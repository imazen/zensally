#!/usr/bin/env python3
"""Matched pair corrections on frozen crop embeddings and weak FIT labels."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from artifact_identity import sha
from distill_attention_linear import Student,write
from local_rank_distillation import family_weights
from pair_relationship_head import PairCorrection
from nested_pair_training import compatibility
from fit_nested_pairs import BASELINE

ARMS=['additive-linear','pair-real','pair-permuted']


def choices(model,hidden,base,eligible,mask):
    logits=model(hidden,base,eligible).masked_fill(~mask,-torch.inf)
    if not mask.any((-2,-1)).all():raise ValueError('every request requires a strict pair')
    flat=logits.flatten(-2).argmax(-1);return torch.stack([flat//128,flat%128],-1)


def weak_metrics(choice,cache,indices):
    flat=choice.reshape(len(indices),22);rows=[]
    for j,i in enumerate(indices):
        for q in range(18):
            if bool(cache['reference_valid'][i,q].any()):
                k=int(flat[j,q]);rows.append(dict(id=cache['sources'][i]['id'],family_id=cache['sources'][i]['family_id'],query=q,selected=k,cut=bool(cache['negative'][i,q,k]),match=bool(cache['positive'][i,q,k])))
    groups={}
    for r in rows:groups.setdefault(r['family_id'],[]).append(r)
    return dict(requests=len(rows),families=len(groups),cut_requests=sum(r['cut'] for r in rows),matched_requests=sum(r['match'] for r in rows),**{key:sum(sum(r[key] for r in g)/len(g) for g in groups.values())/len(groups) for key in ['cut','match']}),rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old','rank','new','initializer','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    expected=[(a.old,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.initializer,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered pair inputs differ')
    old=torch.load(a.old,map_location='cpu',weights_only=False);rank=torch.load(a.rank,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);state=torch.load(a.initializer,map_location='cpu',weights_only=False)['model']
    if old['sources']!=rank['sources']:raise ValueError('original hidden/native identity differs')
    both=old['sources']+new['sources']
    if any(s['role'] not in ['fit','dev'] for s in both) or {s['family_id'] for s in both if s['role']=='fit'}&{s['family_id'] for s in both if s['role']=='dev'}:raise ValueError('fixed disjoint FIT/DEV only')
    fit=[i for i,s in enumerate(old['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(old['sources']) if s['role']=='dev'];ndev=[i for i,s in enumerate(new['sources']) if s['role']=='dev']
    if list(map(len,[fit,dev,ndev]))!=[94,25,24]:raise ValueError('registered roles differ')
    scalar=Student({k:state['base.'+k] for k in ['weight','bias']},state['teacher'],state['mean'],state['std'],nonlinear=True);scalar.load_state_dict(state);scalar.requires_grad_(False)
    with torch.no_grad():base=scalar(old['hidden'],None).reshape(119,11,2,128);nbase=scalar(new['hidden'],None).reshape(72,11,2,128)
    hidden=old['hidden'].reshape(119,11,2,128,96);nhidden=new['hidden'].reshape(72,11,2,128,96);eligible=old['eligible'].reshape(119,11,2,128);neligible=new['eligible'].reshape(72,11,2,128)
    scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in rank['sources']])[:,None,None,:];boxes=(rank['boxes']*scale).round().long();mask=compatibility(boxes,eligible)
    ns=torch.tensor([[*s['native_size'],*s['native_size']] for s in new['sources']])[:,None,None,:];nb=(new['query_boxes']*ns).round().long();o=nb[:,::2,:,None,:];i=nb[:,1::2,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(i[...,2]-i[...,0])*(i[...,3]-i[...,1]);nmask=(o[...,:2]<=i[...,:2]).all(-1)&(o[...,2:]>=i[...,2:]).all(-1)&(ia<oa)&neligible[...,0,:,None]&neligible[...,1,None,:]
    if not mask.any((-2,-1)).all() or not nmask.any((-2,-1)).all():raise ValueError('frozen pair banks require full feasibility')
    valid=old['query_valid'].reshape(119,11,2).all(-1);target=old['targets'].reshape(119,11,2,128);phase=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%11 for s in old['sources']]);fw=family_weights(old['sources'],fit)*len(fit);weight_lookup={i:fw[j] for j,i in enumerate(fit)}
    def regret(choice):
        selected=choice.reshape(25,22);t=old['targets'][dev];top=t.masked_fill(~old['eligible'][dev],-torch.inf).argmax(-1);r=(t.gather(-1,top[...,None])-t.gather(-1,selected[...,None]))[...,0];v=old['query_valid'][dev,:18]
        return float((((r[:,:18]*v).sum(-1)/v.sum(-1))*family_weights(old['sources'],dev)).sum())
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},arms=ARMS,seeds=[31,137,911],epochs=32,optimizer='AdamW0.001 decay0.0001 clip5 FP32 CPU two threads',loss='temperature0.5 pair target sum listwise KL; FIT paired-valid only',raw_endpoint_frozen=True,selection='unchanged old paired essential gate then regret then earliest epoch',new_dev_selection=False,api_calls=0,test_inference=False,model_promoted=False))
    status=dict(pid=os.getpid(),stage='fitting',completed=0,required=9)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[]
    for arm in ARMS:
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=PairCorrection(arm,state['mean'],state['std']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);directory=a.output/f'{arm}-seed{seed}';directory.mkdir();history=[];best=None;fallback=None;selected=None;failed=None
            for epoch in range(33):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),8):
                        ix=torch.tensor(order[start:start+8]);ri=(phase[ix]+epoch)%11;keep=valid[ix,ri];ix=ix[keep];ri=ri[keep]
                        if not len(ix):continue  # Explicitly unknown paired targets supply no gradient.
                        optimizer.zero_grad(set_to_none=True);logits=model(hidden[ix,ri],base[ix,ri],eligible[ix,ri]);m=mask[ix,ri];t=target[ix,ri];teacher=t[:,0,:,None].double()+t[:,1,None,:].double();logp=F.log_softmax((logits/.5).masked_fill(~m,-1e30).flatten(1),-1);prob=F.softmax((teacher/.5).masked_fill(~m,-1e30).flatten(1),-1);loss=F.kl_div(logp,prob,reduction='none').sum(-1);w=torch.stack([weight_lookup[int(i)] for i in ix]);total=(loss*w).sum()/w.sum()
                        if not torch.isfinite(total):raise ValueError('nonfinite pair relationship loss')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                with torch.no_grad():
                    choice=choices(model,hidden[dev],base[dev],eligible[dev],mask[dev]);value=regret(choice);nc=choices(model,nhidden[ndev],nbase[ndev],neligible[ndev],nmask[ndev]);metrics,rows=weak_metrics(nc,new,ndev)
                    if epoch==0 and value!=.2601603865623474:raise ValueError('common initial pair choices differ')
                record=dict(epoch=epoch,paired_regret=value,old_gate=value<=BASELINE,metrics=metrics);history.append(record);path=directory/f'epoch{epoch:02d}.pt';torch.save(dict(record,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),old_choices=choice,new_choices=nc,rows=rows),path);key=(value,epoch)
                if fallback is None or key<fallback:fallback=key;failed=path;failed_record=record
                if record['old_gate'] and (best is None or key<best):best=key;selected=path;selected_record=record
            qualifies=selected is not None;chosen=selected if qualifies else failed;record=selected_record if qualifies else failed_record;shutil.copy2(chosen,directory/('selected.pt' if qualifies else 'best-ineligible.pt'));write(directory/'history.json',history);reports.append(dict(arm=arm,seed=seed,qualified=qualifies,checkpoint_sha256=sha(chosen),**record));write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    if not torch.equal(base,scalar(old['hidden'],None).reshape_as(base)):raise ValueError('raw endpoint scorer changed')
    progress(pid=None,stage='complete',completed=9);print(json.dumps(reports))


if __name__=='__main__':main()
