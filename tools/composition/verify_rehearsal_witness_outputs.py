#!/usr/bin/env python3
"""Bind displayed cached-head choices to complete neural serving decisions."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from artifact_identity import sha
from export_unsplash_ranker import LocalCropRanker,session
from distill_attention_linear import Student
from evaluate_attention_fallback import StudentScorer


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['gallery','cache','weights','control','serving']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df' or sha(a.control/'SHA256SUMS.json')!='941d64dca4c3f0cf5477fdd78ac332dee1a080818c825e44b7c8f17043bb6fc4':raise ValueError('frozen witness parents differ')
    data=torch.load(a.cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(data['sources'])};initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];base=LocalCropRanker().cuda().eval();base.load_state_dict(initial);models={'frozen-original':base};identities={}
    for arm in ['replay-only','replay-contrastive','real','permuted-target']:
        path=a.control/'relationship-rehearsal'/(arm+'-seed31')/'selected.pt';s=torch.load(path,map_location='cpu',weights_only=False)['model'];model=LocalCropRanker().cuda().eval();model.load_state_dict(initial)
        student=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student.load_state_dict(s);model.ranker.scorer[2]=StudentScorer(student).cuda().eval();models[arm]=model;identities[arm]=sha(path)
    qualification=json.loads((a.serving/'qualification.json').read_text())
    for name in ['encoder','baseline-query','candidate-query']:
        if sha(a.serving/(name+'.onnx'))!=qualification['exports'][name]['sha256']:raise ValueError('serving graph changed')
    encoder=session(a.serving/'encoder.onnx');queries={name:session(a.serving/(name+'-query.onnx')) for name in ['baseline','candidate']};rows=[]
    for witness in json.loads((a.gallery/'summary.json').read_text())['witnesses']:
        i=lookup[witness['id']];q=witness['query'];source=data['sources'][i]
        if source['role']!='dev' or source['family_id']!=witness['family_id']:raise ValueError('witness role/family differs')
        image=data['image'][i:i+1];features=base.feature_maps(image.cuda());f=encoder.run(None,dict(image=image.numpy()));p,r=map(int,witness['ratio'].split(':'));boxes=data['query_boxes'][i:i+1,q]
        values=(boxes,torch.tensor([source['native_size'][0]/source['native_size'][1]]),torch.tensor([p/r]),torch.tensor([q%2],dtype=torch.float32),data['content'][i:i+1]);mask=data['eligible'][i,q].numpy();coordinates=boxes.numpy()[0]
        expected={row['arm']:row['weak_record']['selected'] for row in witness['images'] if row['weak_record']}
        def check(backend,arm,scores):
            if not np.isfinite(scores).all():raise ValueError('nonfinite serving output')
            choice=int(np.where(mask,scores,-np.inf).argmax());old=expected[arm];movement=float(np.abs(coordinates[choice]-coordinates[old]).max()*128)
            rows.append(dict(id=witness['id'],query=q,backend=backend,arm=arm,displayed=old,serving=choice,edge_movement_128_grid=movement))
        for arm,model in models.items():check('torch-cuda',arm,model.scores(features,*(v.cuda() for v in values)).cpu().numpy()[0])
        feed=dict(f8=f[0],f16=f[1],**{key:v.numpy() for key,v in zip(['boxes','source_aspect','target_aspect','mode','content'],values)})
        for graph,arm in [('baseline','frozen-original'),('candidate','replay-contrastive')]:check('ort',arm,queries[graph].run(None,feed)[0][0])
    result=dict(rows=rows,checkpoint_sha256=identities,decisions=len(rows),changed_indices=sum(r['displayed']!=r['serving'] for r in rows),max_edge_movement_128_grid=max(r['edge_movement_128_grid'] for r in rows),gate_violations=sum(r['edge_movement_128_grid']>3 for r in rows),api_calls=0,test_inference=False,model_promoted=False)
    with (a.gallery/'serving-check.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','checkpoint_sha256']}))
    if result['gate_violations']:raise SystemExit(1)


if __name__=='__main__':main()
