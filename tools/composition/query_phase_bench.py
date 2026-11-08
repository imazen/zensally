#!/usr/bin/env python3
"""Isolate image analysis and already-cached candidate queries without Torch."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from artifact_identity import sha
from cached_focalnet import session
from query_selection import eligible


def fnv_update(value, choice):
    for byte in int(choice).to_bytes(4,'little'):
        value=((value^byte)*1099511628211)&0xffffffffffffffff
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',type=Path);parser.add_argument('fixture',type=Path)
    parser.add_argument('--phase',choices=['analysis','query'],required=True)
    parser.add_argument('--threads',type=int,choices=[1,2,4],default=1)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--no-spin',action='store_true')
    args=parser.parse_args()
    if not 1<=args.repeats<=1000:raise ValueError('bounded repeats required')
    manifest=json.loads((args.fixture/'manifest.json').read_text())
    selected=manifest['sources']
    if args.phase=='analysis':
        lookup={s['id']:s for s in selected}
        selected=[lookup[source_id] for source_id in manifest['analysis_input_witnesses']]
        if len(selected)!=6 or any(s['image_file'] is None for s in selected):
            raise ValueError('all six declared analysis witnesses required')
    graph=session(args.split/(args.phase+'.onnx'),threads=args.threads,spinning=not args.no_spin)
    sources=[]
    for source in selected:
        feature_file=args.fixture/source['feature_file']
        if sha(feature_file)!=source['feature_sha256']:raise ValueError('source feature input changed')
        features=np.load(feature_file,allow_pickle=False)
        content=np.array([source['content']],np.float32)
        bank_file=args.fixture/source['bank_file']
        if sha(bank_file)!=source['bank_sha256']:raise ValueError('source bank changed')
        bank=json.loads(bank_file.read_text());tasks=[]
        if args.phase=='analysis':
            image_file=args.fixture/source['image_file']
            if sha(image_file)!=source['image_sha256']:raise ValueError('analysis pixels changed')
            image=np.load(image_file,allow_pickle=False);binding=graph.io_binding();binding.bind_cpu_input('image',image)
            outputs={value.name:np.empty(value.shape,np.float32) for value in graph.get_outputs()}
            for name,value in outputs.items():binding.bind_output(name,'cpu',0,np.float32,value.shape,value.ctypes.data)
            sources.append(dict(id=source['id'],binding=binding,outputs=outputs,image=image,expected=features))
        else:
            for task in bank['tasks']:
                boxes=np.zeros((1,128,4),np.float32)
                for j,c in enumerate(task['candidates']):boxes[0,j]=c['box']
                output=np.empty((1,128),np.float32);binding=graph.io_binding()
                for name,value in [('boxes',boxes),('content',content),('transpose',features)]:binding.bind_cpu_input(name,value)
                binding.bind_output('crop_scores','cpu',0,np.float32,output.shape,output.ctypes.data)
                masks=[np.flatnonzero(eligible(task,bank['width'],bank['height'],mode)) for mode in ['minimal','maximal']]
                tasks.append(dict(binding=binding,output=output,boxes=boxes,masks=masks))
            sources.append(dict(id=source['id'],features=features,content=content,tasks=tasks))
    def run(verify=False):
        digest=14695981039346656037
        for source in sources:
            if args.phase=='analysis':
                graph.run_with_iobinding(source['binding'])
                if verify and not np.array_equal(source['outputs']['transpose'].view(np.uint32),source['expected'].view(np.uint32)):
                    raise ValueError('analysis frontier changed bits')
            else:
                for task in source['tasks']:
                    graph.run_with_iobinding(task['binding'])
                    for indices in task['masks']:
                        choice=int(indices[np.argmax(task['output'][0,indices])]) if len(indices) else 0xffffffff
                        digest=fnv_update(digest,choice)
        return digest
    digest=run(verify=True)
    requests=sum(len(s['tasks'])*2 for s in sources) if args.phase=='query' else len(sources)
    if args.phase=='query' and str(digest)!=manifest['reference_choice_fnv64']:
        raise ValueError('cached query selections differ from the whole-bank gate')
    start=time.perf_counter_ns()
    for _ in range(args.repeats):run()
    elapsed=time.perf_counter_ns()-start
    print('CHECK '+json.dumps(dict(digest=str(digest),requests=requests)))
    print('RESULT '+json.dumps(dict(inference_ns=elapsed/(args.repeats*len(sources)),
        phase=args.phase,sources=len(sources),threads=args.threads,spinning=not args.no_spin,
        os_thread_count=len(os.listdir('/proc/self/task')),
        scope='analysis: normalized RGB to reused frontier/importance outputs; query: preloaded neural feature inputs to 11 query scores and fixed-mask selections/source; excludes CNN, image preprocessing and IO for query phase')))


if __name__=='__main__':main()
