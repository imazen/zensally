#!/usr/bin/env python3
"""Frozen C002 hidden values on the source-frozen broader people DEV banks."""
import argparse
import atexit
import os
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from artifact_identity import sha
from fit_unsplash_ranks import encode
from local_rank_distillation import LocalCropRanker
from distill_attention_linear import write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','broad','weights','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fixed=[(a.plan,'1b2da79fe0a04d6a73121529d9e2c8940052cfb1812ccadcfc0fd70f01814e4a'),(a.broad,'420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d'),(a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df')]
    if any(sha(p)!=h for p,h in fixed):raise ValueError('frozen broader inputs differ')
    sources=json.loads(a.plan.read_text())['sources'];full=torch.load(a.broad,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(full['sources'])};indices=[lookup[s['id']] for s in sources]
    for source,i in zip(sources,indices):
        if any(source[k]!=full['sources'][i][k] for k in ['id','role','family_id','native_size','working_size','png_sha256','rgb_sha256']) or source['role']!='dev':raise ValueError('fixed source role/pixels differ')
    a.output.mkdir(parents=True,exist_ok=False)
    terminal=[False]
    def failed():
        if not terminal[0]:write(a.output/'PROGRESS.json',dict(pid=None,stage='failed-artifacts-retained'))
    atexit.register(failed)
    write(a.output/'PROGRESS.json',dict(pid=os.getpid(),stage='extracting',completed=0,required=22))
    model=LocalCropRanker().cuda();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model']);model.eval();images=full['image'][indices].cuda();maps=encode(model,images,list(range(16)));content=full['content'][indices].cuda();aspect=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda');hidden=[];scores=[];captured=[];hook=model.ranker.scorer[2].register_forward_pre_hook(lambda module,args:captured.append(args[0]));query_boxes=full['boxes'][indices].repeat_interleave(2,1)
    for q in range(22):
        p,r=map(int,full['ratios'][q//2].split(':'));raw=model.scores(maps,query_boxes[:,q].cuda(),aspect,aspect.new_full((16,),p/r),aspect.new_full((16,),q%2),content)
        if len(captured)!=1:raise ValueError('one frozen hidden capture required')
        value=captured.pop()
        if value.shape!=(16,128,96) or not torch.equal(raw,F.linear(value,model.ranker.scorer[2].weight,model.ranker.scorer[2].bias)[...,0]):raise ValueError('C002 head replay differs')
        hidden.append(value.cpu());scores.append(raw.cpu());write(a.output/'PROGRESS.json',dict(pid=os.getpid(),stage='extracting',completed=q+1,required=22))
    hook.remove();scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in sources])[:,None,None,:];native=(query_boxes*scale).round().long()
    for j,i in enumerate(indices):
        for ri in range(11):
            expected=torch.tensor(full['native_boxes'][i][ri])
            if not torch.equal(native[j,ri*2,:len(expected)],expected):raise ValueError('cached bank native coordinates differ')
    result=dict(schema='broader-people-hidden-v1',sources=sources,ratios=full['ratios'],hidden=torch.stack(hidden,1),initial_scores=torch.stack(scores,1),query_boxes=query_boxes,**{k:full[k][indices] for k in ['eligible','targets','query_valid','teacher_scores','pairs','pair_valid']},identities={str(p):h for p,h in fixed},api_calls=0,test_inference=False,trained=False)
    if not torch.isfinite(result['hidden']).all():raise ValueError('nonfinite hidden values')
    torch.save(result,a.output/'cache.pt');write(a.output/'summary.json',dict(sources=16,hidden_shape=list(result['hidden'].shape),cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,head_replay_exact=True,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),api_calls=0,test_inference=False));write(a.output/'PROGRESS.json',dict(pid=None,stage='complete',completed=22,required=22));terminal[0]=True;print(json.dumps(json.loads((a.output/'summary.json').read_text())))


if __name__=='__main__':main()
