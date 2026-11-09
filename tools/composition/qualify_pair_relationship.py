#!/usr/bin/env python3
"""Full image-network CUDA qualification for fixed-endpoint pair corrections."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from fit_unsplash_ranks import encode,LocalCropRanker
from local_rank_distillation import family_weights
from pair_relationship_head import PairCorrection
from fit_pair_relationship import choices,weak_metrics,ARMS
from nested_pair_training import compatibility
from fit_nested_pairs import BASELINE


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','old','new','new_hidden','weights','endpoint','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected=[(a.old,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9'),(a.new_hidden,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'),(a.endpoint,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered full inference inputs differ')
    for name in ['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):raise ValueError('frozen quality/model source differs')
    fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if len(fits)!=9 or {(r['arm'],r['seed']) for r in fits}!={(arm,seed) for arm in ARMS for seed in [31,137,911]}:raise ValueError('registered nine fits required')
    s=torch.load(a.endpoint,map_location='cpu',weights_only=False)['model'];endpoint=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);endpoint.load_state_dict(s);endpoint.cuda().eval();network=LocalCropRanker().cuda();network.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model']);network.eval();cohorts={}
    for cohort,path in [('old',a.old),('new',a.new)]:
        c=torch.load(path,map_location='cpu',weights_only=False);indices=[i for i,v in enumerate(c['sources']) if v['role']=='dev']
        if len(indices)!=(25 if cohort=='old' else 24) or any(v['role'] not in ['fit','dev'] for v in c['sources']):raise ValueError('frozen role coverage differs')
        images=c['image'][indices].cuda();features=encode(network,images,list(range(len(indices))));content=c['content'][indices].cuda();aspects=torch.tensor([c['sources'][i]['native_size'][0]/c['sources'][i]['native_size'][1] for i in indices],device='cuda');hidden=[];captured=[]
        hook=network.ranker.scorer[2].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        for q in range(22):
            p,r=map(int,c['ratios'][q//2].split(':'));b=c['boxes'][indices,q//2] if cohort=='old' else c['query_boxes'][indices,q];network.scores(features,b.cuda(),aspects,torch.full_like(aspects,p/r),torch.full_like(aspects,q%2),content)
            if len(captured)!=1:raise ValueError('one frozen hidden capture required')
            hidden.append(captured.pop())
        hook.remove();h=torch.stack(hidden,1).reshape(len(indices),11,2,128,96);base=endpoint(h,None);e=c['eligible'][indices].reshape(len(indices),11,2,128);scale=torch.tensor([[*c['sources'][i]['native_size'],*c['sources'][i]['native_size']] for i in indices])[:,None,None,:]
        native=(c['boxes'][indices]*scale).round().long() if cohort=='old' else (c['query_boxes'][indices]*scale).round().long();boxes=native[:,:,None].expand(-1,-1,2,-1,-1) if cohort=='old' else native.reshape(len(indices),11,2,128,4)
        if cohort=='old':mask=compatibility(native,e)
        else:
            o=boxes[:,:,0,:,None,:];inner=boxes[:,:,1,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);mask=(o[...,:2]<=inner[...,:2]).all(-1)&(o[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&e[...,0,:,None]&e[...,1,None,:]
        cohorts[cohort]=dict(cache=c,indices=indices,hidden=h,base=base,eligible=e,mask=mask,boxes=boxes,scale=scale)
    a.output.mkdir(parents=True,exist_ok=False);results=[];nh=torch.load(a.new_hidden,map_location='cpu',weights_only=False)
    for entry in fits:
        arm=entry['arm'];seed=entry['seed'];path=a.fits/f'{arm}-seed{seed}'/('selected.pt' if entry['qualified'] else 'best-ineligible.pt');state=torch.load(path,map_location='cpu',weights_only=False);model=PairCorrection(arm,s['mean'],s['std']);model.load_state_dict(state['model']);model.cuda().eval();row=dict(arm=arm,seed=seed,epoch=entry['epoch'],checkpoint_sha256=sha(path),cohorts={})
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('chosen checkpoint differs')
        for cohort,d in cohorts.items():
            actual=choices(model,d['hidden'],d['base'],d['eligible'].cuda(),d['mask'].cuda()).cpu();expected_choice=state['old_choices' if cohort=='old' else 'new_choices'];movement=[]
            for j in range(len(d['indices'])):
                for ri in range(11):
                    for mode in range(2):
                        one=d['boxes'][j,ri,mode,actual[j,ri,mode]];other=d['boxes'][j,ri,mode,expected_choice[j,ri,mode]];movement.append(float(((one-other).abs().float()/d['scale'][j,0,0]*128).max()))
            numeric=dict(pair_requests=len(d['indices'])*11,endpoint_changes=int((actual!=expected_choice).sum()),gate_violations=sum(v>3 for v in movement),max_edge_movement_grid=max(movement));row['cohorts'][cohort]=numeric
            if cohort=='old':
                c=d['cache'];ix=d['indices'];t=c['targets'][ix];e=c['eligible'][ix];top=t.masked_fill(~e,-torch.inf).argmax(-1);flat=actual.reshape(25,22);regret=(t.gather(-1,top[...,None])-t.gather(-1,flat[...,None]))[...,0];valid=c['query_valid'][ix,:18];fw=family_weights(c['sources'],ix);value=float((((regret[:,:18]*valid).sum(-1)/valid.sum(-1))*fw).sum());row['full_paired_regret']=value;row['original_pair_gate']=value<=BASELINE
                raw=d['base'].cpu().reshape(25,22,128).masked_fill(~e,-torch.inf).argmax(-1);r=(t.gather(-1,top[...,None])-t.gather(-1,raw[...,None]))[...,0];row['full_raw_regret']=float((((r[:,:18]*valid).sum(-1)/valid.sum(-1))*fw).sum());row['original_both_gates']=row['original_pair_gate'] and row['full_raw_regret']<=BASELINE
            else:row['new_weak'],new_rows=weak_metrics(actual,nh,d['indices']);write(a.output/f'{arm}-seed{seed}-new-predictions.json',new_rows)
        results.append(row)
    write(a.output/'summary.json',dict(inputs={str(path):digest for path,digest in expected},rows=results,api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(results))
    if any(c['gate_violations'] for r in results for c in r['cohorts'].values()):raise SystemExit('three-grid numerical gate failed')


if __name__=='__main__':main()
