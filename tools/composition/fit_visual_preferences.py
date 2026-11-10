#!/usr/bin/env python3
"""Matched bounded residual adaptation on explicit FIT-only weak visual pairs."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import torch
from torch.nn import functional as F
from artifact_identity import sha
from distill_attention_linear import Student, write
from fit_relationship_rehearsal import old_metrics
from fit_attention_scalar import evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import objective, family_weights
from acceptable_crop_sets import acceptable_set_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old','new','control','review','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sha(a.old)!='41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af' or sha(a.new)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered hidden caches differ')
    weights=a.control/'relationship-rehearsal/replay-contrastive-seed31/selected.pt'
    if sha(weights)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b':raise ValueError('common initializer differs')
    old=torch.load(a.old,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False)
    initial=torch.load(weights,map_location='cpu',weights_only=False)['model'];lookup={s['id']:i for i,s in enumerate(new['sources'])}
    fitold=[i for i,s in enumerate(old['sources']) if s['role']=='fit'];devold=[i for i,s in enumerate(old['sources']) if s['role']=='dev'];fitnew=[i for i,s in enumerate(new['sources']) if s['role']=='fit'];devnew=[i for i,s in enumerate(new['sources']) if s['role']=='dev']
    if list(map(len,[fitold,devold,fitnew,devnew]))!=[94,25,48,24]:raise ValueError('fixed source roles differ')
    both=old['sources']+new['sources']
    if {s['family_id'] for s in both if s['role']=='fit'}&{s['family_id'] for s in both if s['role']=='dev'}:raise ValueError('family crosses roles')
    labels_path=a.review/'weak-preferences.json';labels=json.loads(labels_path.read_text());cases=json.loads((a.review/'summary.json').read_text())['cases'];preferences=labels['preferences'];review_hidden=[];review_weights=[]
    if len(preferences)!=6 or labels['excellent_intervals_admitted'] or labels['continuous_crop_labels_admitted']:raise ValueError('only registered six weak comparisons admitted')
    for row in preferences:
        i=lookup[row['id']];source=new['sources'][i];case=cases[row['case']];q=row['query']
        if source['role']!='fit' or row['role']!='fit' or source['family_id']!=row['family_id'] or case['id']!=row['id'] or not row['research_training_only'] or row['absolute_quality_label']:raise ValueError('weak FIT label identity differs')
        for key,field in [('preferred','preferred_image_sha256'),('dispreferred','dispreferred_image_sha256')]:
            candidate=row[key]
            if not bool(new['eligible'][i,q,candidate]):raise ValueError('review candidate ineligible')
            matched=[]
            for pane in case['panes']:
                query=case['query']+(pane['name']=='raw-maximal' or pane['name'].endswith('-inner'))
                if query==q and pane['record']['candidate']==candidate and pane['sha256']==row[field]:matched.append(pane)
            if not matched or any(sha(Path(p['path']))!=row[field] for p in matched):raise ValueError('reviewed image identity differs')
        review_hidden.append(new['hidden'][i,q,[row['preferred'],row['dispreferred']]])
        review_weights.append(row['weight'])
    review_hidden=torch.stack(review_hidden);review_weights=torch.tensor(review_weights)
    def make():
        model=Student(new['scorer_state'],initial['teacher'],initial['mean'],initial['std'],nonlinear=True);model.load_state_dict(initial);return model
    def pair_loss(model,reverse=False):
        scores=model(review_hidden,None);delta=scores[:,1]-scores[:,0]
        return (F.softplus(-delta if reverse else delta)*review_weights).sum()/review_weights.sum(),scores
    phaseold=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%22 for s in old['sources']]);phasenew=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%22 for s in new['sources']]);fw=family_weights(old['sources'],fitold)*len(fitold);weight_lookup={i:fw[j] for j,i in enumerate(fitold)}
    baseline_model=make();baseline_old,_=old_metrics(baseline_model,old,devold);baseline_pair,_=pair_loss(baseline_model)
    a.output.mkdir(parents=True,exist_ok=False);arms=['replay','review','reversed-review'];status=dict(pid=os.getpid(),stage='fitting',completed=0,required=9)
    write(a.output/'protocol.json',dict(inputs={str(a.old):sha(a.old),str(a.new):sha(a.new),str(weights):sha(weights),str(labels_path):sha(labels_path)},arms=arms,seeds=[31,137,911],epochs=16,batch_old=8,batch_new=8,optimizer='AdamW lr0.0003 decay0.0001 clip5 FP32 CPU two threads',old_initializer_regret=baseline_old,initial_review_loss=float(baseline_pair),review_weight=.05,selection='old essential CPU regret <= common checkpoint epoch zero; canonical FIT review loss, old regret, earliest epoch; new DEV diagnostic only; final full-model original CUDA gate required',api_calls=0,test_inference=False,model_promoted=False))
    def progress(**kwargs):status.update(kwargs,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[]
    for arm in arms:
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=make();optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.0003,weight_decay=.0001);directory=a.output/f'{arm}-seed{seed}';directory.mkdir();history=[];best=None;selected_path=None
            for epoch in range(17):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fitold)[torch.randperm(len(fitold),generator=torch.Generator().manual_seed(seed+epoch))].tolist();new_order=torch.tensor(fitnew)[torch.randperm(len(fitnew),generator=torch.Generator().manual_seed(seed+epoch+1000))].tolist()
                    for start in range(0,len(order),8):
                        ix=order[start:start+8];jx=[new_order[(start+k)%len(new_order)] for k in range(8)];optimizer.zero_grad(set_to_none=True)
                        q=(phaseold[ix]+epoch)%22;scores=model(old['hidden'][ix,q],None);loss=objective(scores,old['targets'][ix,q],old['pairs'][ix,q],old['pair_valid'][ix,q],old['eligible'][ix,q],old['query_valid'][ix,q]);valid=old['query_valid'][ix,q];weights_old=torch.stack([weight_lookup[i] for i in ix]);total=(loss*weights_old).sum()/(valid*weights_old).sum().clamp_min(1)
                        q=(phasenew[jx]+epoch)%22;scores=model(new['hidden'][jx,q],None);loss,valid=acceptable_set_loss(scores,new['positive'][jx,q],new['negative'][jx,q]);total=total+.25*(loss*valid).sum()/valid.sum().clamp_min(1)
                        if arm!='replay':total=total+.05*pair_loss(model,reverse=arm=='reversed-review')[0]
                        if not torch.isfinite(total):raise ValueError('nonfinite adaptation objective')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
                        if any(not torch.equal(v,new['scorer_state'][k]) for k,v in model.base.state_dict().items()):raise ValueError('frozen C002 scalar changed')
                with torch.no_grad():
                    regret,choices=old_metrics(model,old,devold);metrics,rows,scores=evaluate(model,new['hidden'],new['extras'],new,devnew);review_loss,review_scores=pair_loss(model);accuracy=float((review_scores[:,0]>review_scores[:,1]).float().mean())
                record=dict(epoch=epoch,old_regret=regret,old_gate=regret<=baseline_old,review_loss=float(review_loss),review_accuracy=accuracy,metrics=metrics);history.append(record)
                state=dict(record,model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),old_choices=choices,rows=rows,scores=scores,review_scores=review_scores,baseline_old=baseline_old,fused=None,teacher_sha256=sha(weights));path=directory/f'epoch{epoch:02d}.pt'
                if path.exists():raise ValueError('refuse to overwrite retained epoch')
                torch.save(state,path)
                key=(float(review_loss),regret,epoch)
                if record['old_gate'] and (best is None or key<best):best=key;selected_path=path;selected=dict(arm=arm,seed=seed,selected_epoch=epoch,old_regret=regret,baseline_old=baseline_old,review_loss=float(review_loss),review_accuracy=accuracy,metrics=metrics)
            if selected_path is None:raise ValueError('epoch zero quality gate unexpectedly unavailable')
            shutil.copy2(selected_path,directory/'selected.pt');write(directory/'history.json',history);reports.append(selected);write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    progress(pid=None,stage='complete',completed=9);print(json.dumps(reports))


if __name__=='__main__':main()
