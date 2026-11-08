#!/usr/bin/env python3
"""Paired multi-ratio workload: repeated full graph versus one cached analysis."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cached_focalnet import CachedContext, prepared, session
from focalnet_bench import IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path)
    parser.add_argument('split', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--backend', choices=['full','cached'], required=True)
    parser.add_argument('--threads', type=int, choices=[1,2,4], default=1)
    parser.add_argument('--query-threads', type=int, choices=[1,2,4])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        raise ValueError('bounded repeats required')
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.models/'focalnet-source/src'))
    from focalnet.imaging import Letterbox, restore_map
    from focalnet.crop import Crop, score_crop
    auxiliary = torch.load(args.cache, map_location='cpu', weights_only=False)
    indices = {s['id']:i for i,s in enumerate(auxiliary['sources'])}
    inputs = []
    for source_id in IDS:
        index = indices[source_id]
        bank = json.loads((args.bank/(source_id+'.json')).read_text())
        image, content = prepared(auxiliary,index)
        tasks = []
        for task in bank['tasks']:
            boxes = np.zeros((1,128,4),np.float32)
            for j,candidate in enumerate(task['candidates']): boxes[0,j]=candidate['box']
            tasks.append((task,boxes))
        inputs.append(dict(id=source_id,image=image,content=content,bank=bank,tasks=tasks))
    if args.backend=='full':
        graph = session(args.models/'focalnet/focalnet-human.onnx',threads=args.threads)
        outputs = [np.empty((1,1,64,64),np.float32),np.empty((1,128),np.float32)]
        bindings = []
        for image in inputs:
            row = []
            for _,boxes in image['tasks']:
                binding = graph.io_binding()
                for name,value in [('image',image['image']),('content',image['content']),('boxes',boxes)]:
                    binding.bind_cpu_input(name,value)
                for name,value in zip(['importance','crop_scores'],outputs):
                    binding.bind_output(name,'cpu',0,np.float32,value.shape,value.ctypes.data)
                row.append(binding)
            bindings.append(row)
    else:
        context = CachedContext(args.split,threads=args.threads,query_threads=args.query_threads)

    checks = []
    def run(collect=False):
        for i,image in enumerate(inputs):
            if args.backend=='cached': token=context.analyze(image['image'],image['content'],image['id'])
            for j,(task,boxes) in enumerate(image['tasks']):
                if args.backend=='full':
                    graph.run_with_iobinding(bindings[i][j]);importance,scores=outputs
                else:
                    scores=context.score(boxes,token);importance=context.outputs['importance']
                if collect:
                    bank=image['bank'];w,h=bank['width'],bank['height'];x0,y0,x1,y1=np.rint(image['content'][0]*256).astype(int)
                    restored=restore_map(importance[0,0],Letterbox(256,x0,y0,x1-x0,y1-y0))
                    retention=[score_crop(restored,Crop(b[0],b[1],b[2]-b[0],b[3]-b[1],0),w,h) for b in [c['pixel_box'] for c in task['candidates']]]
                    p,q=map(int,task['ratio'].split(':'));maximum=min(w//p,h//q)
                    for mode in ['minimal','maximal']:
                        eligible=[n for n,c in enumerate(task['candidates']) if (c['pixel_box'][2]-c['pixel_box'][0]==maximum*p if mode=='minimal' else c['pixel_box'][2]-c['pixel_box'][0]<maximum*p)]
                        if not eligible and mode=='maximal':eligible=list(range(len(task['candidates'])))
                        chosen=None
                        if eligible:
                            best=max(retention[n] for n in eligible);chosen=max((n for n in eligible if retention[n]>=best-.05),key=lambda n:float(scores[0,n]))
                        checks.append([image['id'],task['ratio'],mode,None if chosen is None else task['candidates'][chosen]['pixel_box']])
    run(collect=True)
    start=time.perf_counter_ns()
    for _ in range(args.repeats):run()
    elapsed=time.perf_counter_ns()-start
    digest=hashlib.sha256(json.dumps(checks,separators=(',',':')).encode()).hexdigest()
    print('CHECK '+json.dumps(dict(digest=digest,requests=len(checks))))
    print('RESULT '+json.dumps(dict(inference_ns=elapsed/(args.repeats*len(inputs)),
        backend=args.backend,threads=args.threads,query_threads=args.query_threads or args.threads,
        sources=len(inputs),ratios_per_source=11,
        forwards_per_source=11 if args.backend=='full' else 12,
        scope='warm normalized source to eleven full candidate score arrays; full recomputes encoder eleven times, cached analyzes once then eleven heads; reusable outputs in both; cached validation/input copies included; excludes resize, selection, input load and IO')))


if __name__ == '__main__':main()
