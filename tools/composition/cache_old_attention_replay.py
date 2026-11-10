#!/usr/bin/env python3
"""Freeze original FIT/DEV hidden values for rehearsal with unchanged old labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','weights','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('fixed old parent differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);sources=cache['sources']
    if Counter(s['role'] for s in sources)!=Counter(fit=94,dev=25):raise ValueError('original94/25 roles required')
    model=LocalCropRanker().cuda();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model']);model.eval()
    features=encode(model,cache['image'].cuda(),list(range(119)));aspect=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda');content=cache['content'].cuda();hidden=[];scores=[];captured=[]
    hook=model.ranker.scorer[2].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    for q in range(22):
        p,r=map(int,cache['ratios'][q//2].split(':'))
        raw=model.scores(features,cache['boxes'][:,q//2].cuda(),aspect,aspect.new_full((119,),p/r),aspect.new_full((119,),q%2),content)
        if len(captured)!=1:raise ValueError('unexpected scorer capture count')
        value=captured.pop()
        if not torch.equal(raw,F.linear(value,model.ranker.scorer[2].weight,model.ranker.scorer[2].bias)[...,0]):raise ValueError('original score replay differs')
        hidden.append(value.cpu());scores.append(raw.cpu())
    hook.remove();a.output.mkdir(parents=True,exist_ok=False)
    result=dict(hidden=torch.stack(hidden,1),initial_scores=torch.stack(scores,1),sources=sources,ratios=cache['ratios'],parent_cache_sha256=sha(a.cache),initializer_sha256=sha(a.weights),**{k:cache[k] for k in ['targets','pairs','pair_valid','query_valid','eligible']})
    torch.save(result,a.output/'cache.pt')
    report=dict(cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,hidden_shape=list(result['hidden'].shape),fit_sources=94,dev_sources=25,initial_cuda_replay_exact=True,api_calls=0,trained=False,test_inference=False,model_promoted=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
