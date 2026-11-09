#!/usr/bin/env python3
"""FIT-family-selected low-capacity paired crop controls; DEV is report-only."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import math
import torch
from artifact_identity import sha
from distill_attention_linear import write
from fit_nested_crop_head import audited_predictions,validate,source_mean
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from nested_crop_head import context,subset
from nested_ridge import fractions,native,pca,design,ridge,predict

LAMBDAS=torch.tensor([10.**k for k in range(-4,5)],dtype=torch.float64)


def weights(source_indices):
    unique,counts=source_indices.unique(return_counts=True);lookup={int(i):int(n) for i,n in zip(unique,counts)};return torch.tensor([1/(len(unique)*lookup[int(i)]) for i in source_indices],dtype=torch.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','direct','attention','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=[(a.cache,'3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4'),(a.direct,'2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472'),(a.attention,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3')]
    if any(sha(path)!=digest for path,digest in inputs):raise ValueError('frozen paired inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);d=torch.load(a.direct,map_location='cpu',weights_only=False);att=torch.load(a.attention,map_location='cpu',weights_only=False);sources=c['sources'];ctx=context([s['native_size'] for s in sources],c['ratios']);valid=c['target_valid'][...,0]
    if c['target_valid'].shape[-1]!=1 or int(valid.sum())!=135 or len({s['family_id'] for s in sources})!=72 or any(s['role'] not in ['fit','dev'] for s in sources):raise ValueError('fixed finite same-story targets required')
    sizes=ctx['sizes'][:,None,None].repeat(1,1,1,2);target=(c['targets'][:,:,0]*sizes).round().double();y=fractions(target,ctx);_,roundtrip=native(y,ctx)
    if not torch.equal(roundtrip[valid],target[valid]):raise ValueError('bounded parameter inverse changes admitted native endpoints')
    geometry=torch.tensor([[[sa,ta,sa*sa,sa*ta,ta*ta] for ta in [math.log(int(r.split(':')[0])/int(r.split(':')[1])) for r in c['ratios']]] for sa in [math.log(s['native_size'][0]/s['native_size'][1]) for s in sources]],dtype=torch.float64);eligible=att['eligible'][...,None];summary=((att['extras']*eligible).sum(-2)/eligible.sum(-2).clamp_min(1)).reshape(72,11,2,32).mean(-2).double();visual=d['visual'].double()[:,None].expand(-1,11,-1);fit_mask=valid.clone();fit_mask[[i for i,s in enumerate(sources) if s['role']!='fit']]=False;train_i,train_q=fit_mask.nonzero(as_tuple=True);folds=sorted(set(train_i.tolist()))
    if len(train_i)!=99 or len(folds)!=29:raise ValueError('fixed FIT pair/family count differs')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in inputs},lambdas=LAMBDAS.tolist(),fit_requests=99,fit_families=29,folds='leave one FIT family out; normalization/PCA excludes held family',selection='mean essential held-family max-edge error; ties prefer larger lambda',pca_components=8,api_calls=0,test_inference=False,new_teacher_inference=False,model_promoted=False));reports=[]
    terminal=[False]
    def failed():
        if not terminal[0]:write(a.output/'PROGRESS.json',dict(pid=None,stage='failed-artifacts-retained',completed=len(reports),required=3))
    atexit.register(failed)
    for arm,raw in [('geometry',None),('image-pca8',visual),('attention-pca8',summary)]:
        cv=[]
        for held in folds:
            keep=train_i!=held;ix,q=train_i[keep],train_q[keep];w=weights(ix)
            if any(sources[int(i)]['family_id']==sources[held]['family_id'] for i in ix):raise ValueError('held family entered fitting')
            transform=pca(raw[ix,q],w) if raw is not None else None;tx=design(geometry[ix,q],raw[ix,q] if raw is not None else None,transform);linear=ridge(tx,y[ix,q],w,LAMBDAS);vx=design(geometry[held],raw[held] if raw is not None else None,transform);mask=valid[held]&(torch.arange(11)<9)
            for k,value in enumerate(LAMBDAS):
                _,pred=native(predict(vx,linear,k)[None],subset(ctx,[held]));error=((pred[0]-target[held]).abs()*128/ctx['sizes'][held].repeat(2)).amax((-1,-2));secondary=valid[held]&(torch.arange(11)>=9);cv.append(dict(held_id=sources[held]['id'],held_family=sources[held]['family_id'],lambda_value=float(value),primary_requests=int(mask.sum()),edge_error=float(error[mask].mean()) if mask.any() else None,secondary_requests=int(secondary.sum()),secondary_edge_error=float(error[secondary].mean()) if secondary.any() else None))
            write(a.output/'PROGRESS.json',dict(pid=os.getpid(),stage='fit-family-cv',arm=arm,completed_folds=len(cv)//len(LAMBDAS),required_folds=29))
        primary_folds=len({r['held_family'] for r in cv if r['primary_requests']});means={float(value):sum(r['edge_error'] for r in cv if r['lambda_value']==float(value) and r['edge_error'] is not None)/primary_folds for value in LAMBDAS};selected=min(means,key=lambda value:(means[value],-value));w=weights(train_i);transform=pca(raw[train_i,train_q],w) if raw is not None else None;tx=design(geometry[train_i,train_q],raw[train_i,train_q] if raw is not None else None,transform);linear=ridge(tx,y[train_i,train_q],w,torch.tensor([selected],dtype=torch.float64));all_x=design(geometry.flatten(0,1),raw.flatten(0,1) if raw is not None else None,transform);values=predict(all_x,linear).reshape(72,11,5);normalized,boxes=native(values,ctx);validate(boxes,ctx);groups,rows=audited_predictions(boxes,c,d);pair_error=((normalized-c['targets'][:,:,0]).abs()*128).amax((-1,-2));dev_mask=valid.clone();dev_mask[[i for i,s in enumerate(sources) if s['role']!='dev']]=False;dev_mask[:,9:]=False
        def cast(model,device):return {k:v.float().to(device) for k,v in model.items()} if model is not None else None
        numerical=[]
        for device in ['cpu','cuda']:
            g=geometry.float().to(device);r=raw.float().to(device) if raw is not None else None;dx=design(g.flatten(0,1),r.flatten(0,1) if r is not None else None,cast(transform,device));v=predict(dx,cast(linear,device)).reshape(72,11,5);device_ctx=context([s['native_size'] for s in sources],c['ratios'],device=device);_,pred=native(v,device_ctx);validate(pred,device_ctx);pred=pred.cpu();movement=(pred-boxes).abs()*128/ctx['sizes'][:,None,None].repeat(1,1,1,2);check=dict(implementation=device+'-fp32',rectangles=1584,changed_rectangles=int((pred!=boxes).any(-1).sum()),maximum_edge_movement_grid=float(movement.max()),gate_violations=int((movement.amax(-1)>3).sum()))
            if check['gate_violations']:raise ValueError('unchanged numerical edge gate fails')
            numerical.append(check)
        model=dict(transform=transform,linear=linear,lambda_value=selected,arm=arm);torch.save(model,a.output/(arm+'.pt'));write(a.output/(arm+'-cv.json'),cv);write(a.output/(arm+'-predictions.json'),rows);report=dict(arm=arm,lambda_value=selected,fit_family_cv=means,total_folds=len(folds),primary_folds=primary_folds,dev_pair_edge_error=source_mean(pair_error,dev_mask),groups=groups,numerical=numerical,teacher_inputs_at_inference=raw is summary,checkpoint_sha256=sha(a.output/(arm+'.pt')));reports.append(report);write(a.output/'summary.json',reports);print(json.dumps(report))
    write(a.output/'PROGRESS.json',dict(pid=None,stage='complete',completed=3,required=3));terminal[0]=True


if __name__=='__main__':main()
