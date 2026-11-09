#!/usr/bin/env python3
"""Matched replay, endpoint and structured-pair fits on frozen FIT targets."""
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
from fit_relationship_rehearsal import old_metrics
from fit_attention_scalar import evaluate
from local_rank_distillation import objective,family_weights
from acceptable_crop_sets import acceptable_set_loss
from nested_pair_training import compatibility,pair_choices,paired_top_loss

BASELINE=.24700142443180084
ARMS=['replay','independent-top','paired-top']


@torch.no_grad()
def paired_metrics(scores,old,mask,indices):
    choice=pair_choices(scores.reshape(len(indices),11,2,128),mask[indices]).reshape(len(indices),22)
    target=old['targets'][indices];eligible=old['eligible'][indices]
    top=target.masked_fill(~eligible,-torch.inf).argmax(-1)
    regret=(target.gather(-1,top[...,None])-target.gather(-1,choice[...,None]))[...,0]
    valid=old['query_valid'][indices,:18];weights=family_weights(old['sources'],indices)
    value=float((((regret[:,:18]*valid).sum(-1)/valid.sum(-1))*weights).sum())
    return value,choice


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old','rank','new','control','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    expected=[(a.old,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered caches differ')
    weights=a.control/'relationship-rehearsal/replay-contrastive-seed31/selected.pt'
    if sha(weights)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b':raise ValueError('common initializer differs')
    old=torch.load(a.old,map_location='cpu',weights_only=False);rank=torch.load(a.rank,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False)
    identity=lambda data:[(s['id'],s['role'],s['family_id'],s['rgb_sha256']) for s in data['sources']]
    if identity(old)!=identity(rank):raise ValueError('hidden/native identity differs')
    both=old['sources']+new['sources']
    if any(s['role'] not in ['fit','dev'] for s in both):raise ValueError('TEST remains closed')
    if {s['family_id'] for s in both if s['role']=='fit'}&{s['family_id'] for s in both if s['role']=='dev'}:raise ValueError('family crosses roles')
    fitold=[i for i,s in enumerate(old['sources']) if s['role']=='fit'];devold=[i for i,s in enumerate(old['sources']) if s['role']=='dev'];fitnew=[i for i,s in enumerate(new['sources']) if s['role']=='fit'];devnew=[i for i,s in enumerate(new['sources']) if s['role']=='dev']
    if list(map(len,[fitold,devold,fitnew,devnew]))!=[94,25,48,24]:raise ValueError('fixed roles differ')
    scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in rank['sources']])[:,None,None,:]
    boxes=(rank['boxes']*scale).round().long();mask=compatibility(boxes,old['eligible'].reshape(119,11,2,128))
    labels=pair_choices(old['targets'].reshape(119,11,2,128),mask);both_valid=old['query_valid'].reshape(119,11,2).all(-1)
    # DEV targets above are used solely to calculate diagnostics, never gradients.
    if int(both_valid[fitold].sum())!=1024:raise ValueError('registered FIT paired target count differs')
    initial=torch.load(weights,map_location='cpu',weights_only=False)['model']
    def make():
        model=Student(new['scorer_state'],initial['teacher'],initial['mean'],initial['std'],nonlinear=True);model.load_state_dict(initial);return model
    phase=lambda data:torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%11 for s in data['sources']])
    phaseold=phase(old);phasenew=phase(new);fw=family_weights(old['sources'],fitold)*len(fitold);weight_lookup={i:fw[j] for j,i in enumerate(fitold)}
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},initializer_sha256=sha(weights),arms=ARMS,seeds=[31,137,911],epochs=32,batch_old=8,batch_new=8,fit_paired_targets=1024,baseline_old=BASELINE,optimizer='AdamW lr0.0003 decay0.0001 clip5 FP32 CPU two threads',selection='both raw and paired essential old DEV regret <= frozen baseline; then paired regret, raw regret, earliest epoch; failed best retained but ineligible',pair_arithmetic='binary64 pair sum, ascending outer/inner index ties',api_calls=0,test_inference=False,model_promoted=False))
    status=dict(pid=os.getpid(),stage='fitting',completed=0,required=9)
    def progress(**kwargs):status.update(kwargs,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[]
    for arm in ARMS:
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=make();optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.0003,weight_decay=.0001);directory=a.output/f'{arm}-seed{seed}';directory.mkdir();history=[];best=None;best_path=None;fallback=None;fallback_path=None
            for epoch in range(33):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fitold)[torch.randperm(len(fitold),generator=torch.Generator().manual_seed(seed+epoch))].tolist();new_order=torch.tensor(fitnew)[torch.randperm(len(fitnew),generator=torch.Generator().manual_seed(seed+epoch+1000))].tolist()
                    for start in range(0,len(order),8):
                        ix=order[start:start+8];jx=[new_order[(start+k)%len(new_order)] for k in range(8)];optimizer.zero_grad(set_to_none=True)
                        ri=(phaseold[ix]+epoch)%11;q=ri[:,None]*2+torch.arange(2);ix_tensor=torch.tensor(ix)[:,None];scores=model(old['hidden'][ix_tensor,q],None)
                        target=old['targets'][ix_tensor,q];valid=old['query_valid'][ix_tensor,q];eligible=old['eligible'][ix_tensor,q]
                        loss=objective(scores.flatten(0,1),target.flatten(0,1),old['pairs'][ix_tensor,q].flatten(0,1),old['pair_valid'][ix_tensor,q].flatten(0,1),eligible.flatten(0,1),valid.flatten()).reshape(len(ix),2)
                        weight=torch.stack([weight_lookup[i] for i in ix]);total=(loss*weight[:,None]).sum()/(valid*weight[:,None]).sum().clamp_min(1)
                        paired_valid=both_valid[ix,ri]
                        if arm!='replay' and paired_valid.any():
                            if arm=='paired-top':extra=paired_top_loss(scores,mask[ix,ri],labels[ix,ri])
                            else:
                                top=target.masked_fill(~eligible,-torch.inf).argmax(-1)
                                extra=F.cross_entropy(scores.masked_fill(~eligible,-torch.inf).flatten(0,1),top.flatten(),reduction='none').reshape(len(ix),2).sum(-1)
                            total=total+.05*(extra*paired_valid*weight).sum()/(paired_valid*weight).sum().clamp_min(1)
                        nr=(phasenew[jx]+epoch)%11;nq=nr[:,None]*2+torch.arange(2);ni=torch.tensor(jx)[:,None];ns=model(new['hidden'][ni,nq],None);nl,nv=acceptable_set_loss(ns,new['positive'][ni,nq],new['negative'][ni,nq]);total=total+.25*(nl*nv).sum()/nv.sum().clamp_min(1)
                        if not torch.isfinite(total):raise ValueError('nonfinite training loss')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
                        if any(not torch.equal(v,new['scorer_state'][k]) for k,v in model.base.state_dict().items()):raise ValueError('base scalar changed')
                with torch.no_grad():
                    raw,raw_choice=old_metrics(model,old,devold);paired,paired_choice=paired_metrics(model(old['hidden'][devold],None),old,mask,devold);metrics,rows,scores=evaluate(model,new['hidden'],new['extras'],new,devnew)
                record=dict(epoch=epoch,old_regret=raw,paired_regret=paired,old_gate=raw<=BASELINE and paired<=BASELINE,metrics=metrics);history.append(record)
                state=dict(record,model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),old_choices=raw_choice,paired_old_choices=paired_choice,rows=rows,scores=scores,baseline_old=BASELINE,fused=None,teacher_sha256=sha(weights));path=directory/f'epoch{epoch:02d}.pt'
                if path.exists():raise ValueError('refuse to overwrite epoch')
                torch.save(state,path);key=(paired,raw,epoch)
                if fallback is None or key<fallback:fallback=key;fallback_path=path;failed_record=record
                if record['old_gate'] and (best is None or key<best):best=key;best_path=path;selected_record=record
            qualifies=best_path is not None;chosen=best_path if qualifies else fallback_path;chosen_record=selected_record if qualifies else failed_record
            shutil.copy2(chosen,directory/('selected.pt' if qualifies else 'best-ineligible.pt'));write(directory/'history.json',history);reports.append(dict(arm=arm,seed=seed,qualified=qualifies,checkpoint=chosen.name,**chosen_record));write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    progress(pid=None,stage='complete',completed=9);print(json.dumps(reports))


if __name__=='__main__':main()
