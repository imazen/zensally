#!/usr/bin/env python3
"""Matched old-quality rehearsal, acceptable-set and relationship student controls."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from fit_attention_scalar import evaluate
from local_rank_distillation import objective,family_weights
from acceptable_crop_sets import acceptable_set_loss

ARMS=['replay-only','replay-contrastive','real','permuted-target']


@torch.no_grad()
def old_metrics(model,data,indices):
    scores=model(data['hidden'][indices],None);mask=data['eligible'][indices]
    selected=scores.masked_fill(~mask,-torch.inf).argmax(-1)
    targets=data['targets'][indices];teacher=targets.masked_fill(~mask,-torch.inf).argmax(-1)
    regret=(targets.gather(-1,teacher[...,None])-targets.gather(-1,selected[...,None]))[...,0]
    valid=data['query_valid'][indices,:18];weights=family_weights(data['sources'],indices)
    per_source=(regret[:,:18]*valid).sum(-1)/valid.sum(-1)
    if not torch.isfinite(per_source).all():raise ValueError('nonfinite old quality')
    return float((per_source*weights).sum()),selected.cpu()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['old','new','relationships','teachers','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    expected=[(a.old,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.relationships,'916b95282b5ed329eaad9710c737314d888fd4db47e48866af4f9da6dba50d84')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered rehearsal inputs differ')
    old=torch.load(a.old,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);rel=torch.load(a.relationships,map_location='cpu',weights_only=False)
    if rel['source_ids']!=[s['id'] for s in new['sources']]:raise ValueError('relationship source order differs')
    fitold=[i for i,s in enumerate(old['sources']) if s['role']=='fit'];devold=[i for i,s in enumerate(old['sources']) if s['role']=='dev']
    fitnew=[i for i,s in enumerate(new['sources']) if s['role']=='fit'];devnew=[i for i,s in enumerate(new['sources']) if s['role']=='dev']
    if list(map(len,[fitold,devold,fitnew,devnew]))!=[94,25,48,24]:raise ValueError('registered roles differ')
    all_sources=old['sources']+new['sources']
    if {s['family_id'] for s in all_sources if s['role']=='fit'}&{s['family_id'] for s in all_sources if s['role']=='dev'}:raise ValueError('source families cross roles')
    values=new['hidden'][fitnew][new['eligible'][fitnew]];mean=values.mean(0);std=values.std(0,unbiased=False).clamp_min(.001)
    pose=new['extras'][...,:16];raw_real=torch.cat([pose,rel['real']],-1);raw_permuted=torch.cat([pose,rel['permuted']],-1)
    phaseold=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%22 for s in old['sources']]);phasenew=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%22 for s in new['sources']])
    weightsold=family_weights(old['sources'],fitold)*len(fitold);weight_lookup={i:weightsold[j] for j,i in enumerate(fitold)}
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},arms=ARMS,seeds=[31,137,911],epochs=88,batch_old=8,batch_new=8,old_query='source hash phase plus epoch modulo22',new_queries='same phased query for contrastive loss; all22 for feature loss',optimizer='AdamW lr0.001 decay0.0001 clip5 FP32 CPU two threads',objective='unchanged family-weighted old rank objective +0.25 new contrastive source mean +0.1 eligible feature MSE; arm terms explicit',selection='old essential CPU regret <= same-platform epoch-zero baseline, then new essential family head cuts, negative reference match, old regret and earliest epoch; final original full-model CUDA gate required',architecture='frozen C002 base plus96->32 ReLU->32 normalized feature student',api_calls=0,test_inference=False,model_promoted=False))
    status=dict(pid=os.getpid(),stage='fitting',completed=0,required=12)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[]
    for arm in ARMS:
        for seed in [31,137,911]:
            teacher_path=a.teachers/('pose-relations-seed'+str(seed))/'selected.pt';teacher=torch.load(teacher_path,map_location='cpu',weights_only=False)
            target=((raw_permuted if arm=='permuted-target' else raw_real)-teacher['mean'])/teacher['std']
            torch.manual_seed(seed);model=Student(new['scorer_state'],teacher['model']['extra.weight'],mean,std,nonlinear=True)
            optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.001,weight_decay=.0001);directory=a.output/(arm+'-seed'+str(seed));directory.mkdir();history=[];best=None
            baseline_old,_=old_metrics(model,old,devold)
            for epoch in range(89):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    rng=torch.Generator().manual_seed(seed+epoch);orders=[torch.tensor(pool)[torch.randperm(len(pool),generator=rng)].tolist() for pool in [fitold,fitnew]]
                    for start in range(0,len(fitold),8):
                        ix=orders[0][start:start+8];query=(phaseold[ix]+epoch-1)%22
                        optimizer.zero_grad(set_to_none=True)
                        score=model(old['hidden'][ix,query],None)
                        loss=objective(score,old['targets'][ix,query],old['pairs'][ix,query],old['pair_valid'][ix,query],old['eligible'][ix,query],old['query_valid'][ix,query])
                        total=(loss*torch.stack([weight_lookup[i] for i in ix])).mean()
                        if arm!='replay-only':
                            jx=[orders[1][(start+j)%len(fitnew)] for j in range(8)];q=(phasenew[jx]+epoch-1)%22
                            score=model(new['hidden'][jx,q],None);contrastive,valid=acceptable_set_loss(score,new['positive'][jx,q],new['negative'][jx,q])
                            total=total+.25*(contrastive*valid).sum()/valid.sum().clamp_min(1)
                            if arm in ['real','permuted-target']:
                                mask=new['eligible'][jx];error=(model.features(new['hidden'][jx])-target[jx]).square().mean(-1)
                                total=total+.1*((error*mask).sum((1,2))/mask.sum((1,2))).mean()
                        if not torch.isfinite(total):raise ValueError('nonfinite rehearsal loss')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
                        if any(not torch.equal(v,new['scorer_state'][k]) for k,v in model.base.state_dict().items()):raise ValueError('original base changed')
                regret,old_choices=old_metrics(model,old,devold);metrics,rows,scores=evaluate(model,new['hidden'],target,new,devnew)
                eligible=regret<=baseline_old;history.append(dict(epoch=epoch,old_regret=regret,old_gate=eligible,metrics=metrics))
                state=dict(epoch=epoch,model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),metrics=metrics,rows=rows,scores=scores,old_regret=regret,old_choices=old_choices,baseline_old=baseline_old,fused=None,teacher_sha256=sha(teacher_path))
                if epoch==0:torch.save(state,directory/'initial.pt')
                key=(metrics['essential']['cut'],-metrics['essential']['match'],regret,epoch)
                if eligible and (best is None or key<best):
                    best=key;torch.save(state,directory/'selected.pt');selected=dict(arm=arm,seed=seed,selected_epoch=epoch,metrics=metrics,old_regret=regret,baseline_old=baseline_old)
                if epoch==88:torch.save(state,directory/'terminal.pt')
            write(directory/'history.json',history);reports.append(selected);write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    progress(pid=None,stage='complete',completed=12);print(json.dumps(reports))


if __name__=='__main__':main()
