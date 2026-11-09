#!/usr/bin/env python3
"""Warm full encoder/query reference timing on fixed old/new DEV tensors."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from artifact_identity import sha
from bench_unsplash_student import session,binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['old','new','export']:p.add_argument(key,type=Path)
    p.add_argument('--arm',choices=['baseline','candidate'],required=True);p.add_argument('--phase',choices=['one','bank'],required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();torch.set_num_threads(1)
    if not 1<=a.repeats<=100:raise ValueError('bounded repeats required')
    report=json.loads((a.export/'qualification.json').read_text())
    if any(r['gate_violations'] for r in report['totals'].values()) or report['test_inference']:raise ValueError('qualified FIT/DEV only graph required')
    for path in [a.old,a.new]:
        if report['inputs'].get(str(path))!=sha(path):raise ValueError('qualified input cache differs')
    for name in ['encoder',a.arm+'-query']:
        if sha(a.export/(name+'.onnx'))!=report['exports'][name]['sha256']:raise ValueError('qualified graph differs')
    encoder=session(a.export/'encoder.onnx',1);query=session(a.export/(a.arm+'-query.onnx'),1)
    frontier={o.name:np.empty(tuple(o.shape),np.float32) for o in encoder.get_outputs()};output=np.empty((1,128),np.float32);masked=np.empty(128,np.float32);views=[];qs=list(range(22)) if a.phase=='bank' else [0]
    for path,expected in [(a.old,25),(a.new,24)]:
        cache=torch.load(path,map_location='cpu',weights_only=False);count=0
        for i,source in enumerate(cache['sources']):
            if source['role']!='dev':continue
            image=cache['image'][i:i+1].numpy();tasks=[]
            for q in qs:
                p,r=map(int,cache['ratios'][q//2].split(':'));boxes=cache['query_boxes'][i:i+1,q] if 'query_boxes' in cache else cache['boxes'][i:i+1,q//2]
                values=dict(boxes=boxes.numpy(),source_aspect=np.array([source['native_size'][0]/source['native_size'][1]],np.float32),target_aspect=np.array([p/r],np.float32),mode=np.array([q%2],np.float32),content=cache['content'][i:i+1].numpy())
                tasks.append(dict(binding=binding(query,dict(**frontier,**values),dict(scores=output)),values=values,eligible=cache['eligible'][i,q].numpy()))
            views.append(dict(binding=binding(encoder,dict(image=image),frontier),image=image,tasks=tasks));count+=1
        if count!=expected:raise ValueError('fixed DEV source count differs')
    def run():
        digest=14695981039346656037
        for view in views:
            encoder.run_with_iobinding(view['binding'])
            for task in view['tasks']:
                query.run_with_iobinding(task['binding']);masked.fill(-np.inf);np.copyto(masked,output[0],where=task['eligible']);choice=int(masked.argmax());digest=((digest^choice)*1099511628211)&0xffffffffffffffff
        return digest
    expected=run();start=time.perf_counter_ns()
    for _ in range(a.repeats):actual=run()
    elapsed=time.perf_counter_ns()-start
    if actual!=expected:raise ValueError('repeated neural choices differ')
    print('CHECK '+json.dumps(dict(sources=49,requests=49*len(qs),gate_violations=0)))
    print('RESULT '+json.dumps(dict(neural_ms=elapsed/a.repeats/49/1e6,choice_digest=str(actual),scope='one ORT encoder per owned cached256x256 input plus queries and fixed-mask choice; excludes decode/resize/candidate generation/safety guards; reference framework, not native Rust pipeline')))


if __name__=='__main__':main()
