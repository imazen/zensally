#!/usr/bin/env python3
"""Bounded weak cut-classifier controls over frozen measured candidate features."""
import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import time
import torch
from artifact_identity import sha
from distill_attention_linear import write
from local_rank_distillation import family_weights
from candidate_cut_head import ARMS,CutHead,features,balanced_source_loss


@torch.no_grad()
def evaluate(model,x,c,indices):
    logits=model(x[indices]);p=c['positive'][indices];n=c['negative'][indices];loss,valid=balanced_source_loss(logits,p,n);prob=logits.sigmoid();sq=(prob-n.float()).square();dims=(1,2);counts=torch.stack([p.sum(dims),n.sum(dims)],-1);present=counts>0;sums=torch.stack([(sq*p).sum(dims),(sq*n).sum(dims)],-1);brier=((sums/counts.clamp_min(1))*present).sum(-1)/present.sum(-1).clamp_min(1);weight=family_weights(c['sources'],indices)*valid;denom=weight.sum()
    if not bool(denom>0):raise ValueError('evaluation requires known labels')
    positive=logits>=0;known=p|n
    return dict(balanced_logistic=float((loss*weight).sum()/denom),balanced_brier=float((brier*weight).sum()/denom),known_sources=int(valid.sum()),known_candidates=int(known.sum()),cut_candidates=int(n.sum()),noncut_candidates=int(p.sum()),true_cut=int((positive&n).sum()),missed_cut=int((~positive&n).sum()),false_cut=int((positive&p).sum()),true_noncut=int((~positive&p).sum())),logits.cpu()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered attention cache differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);fit=[i for i,s in enumerate(c['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(c['sources']) if s['role']=='dev']
    if len(fit)!=48 or len(dev)!=24 or any(s['role'] not in ['fit','dev'] for s in c['sources']) or {c['sources'][i]['family_id'] for i in fit}&{c['sources'][i]['family_id'] for i in dev}:raise ValueError('fixed disjoint FIT/DEV only')
    if bool((c['positive']&c['negative']).any()) or bool(((c['positive']|c['negative'])&~c['eligible']).any()):raise ValueError('known masks must be disjoint eligible candidates')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(cache_sha256=sha(a.cache),arms=ARMS,seeds=[31,137,911],epochs=32,batch_sources=8,queries='all22',architecture='input->32 ReLU->1',optimizer='AdamW0.001 decay0.0001 clip5 FP32 CPU two threads',labels='existing explicit cut negatives are class1, accepted candidate positives class0; all others unknown',normalization='FIT eligible candidates mean/std floor0.001',selection='minimum DEV family-balanced known-class logistic loss then earliest epoch',crop_guard_threshold=.5,api_calls=0,test_inference=False,model_promoted=False));status=dict(pid=os.getpid(),stage='fitting',completed=0,required=15)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[];fw=family_weights(c['sources'],fit)*len(fit);weight_lookup={i:fw[j] for j,i in enumerate(fit)}
    for arm in ARMS:
        x=features(c,arm);train=x[fit][c['eligible'][fit]];mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(.001);del train
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=CutHead(mean,std);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);directory=a.output/f'{arm}-seed{seed}';directory.mkdir();best=None;selected=None;history=[]
            for epoch in range(33):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),8):
                        ix=order[start:start+8];optimizer.zero_grad(set_to_none=True);logits=model(x[ix]);loss,valid=balanced_source_loss(logits,c['positive'][ix],c['negative'][ix]);w=torch.stack([weight_lookup[i] for i in ix])*valid
                        if not bool(w.sum()>0):raise ValueError('registered batch lacks known labels')
                        total=(loss*w).sum()/w.sum()
                        if not torch.isfinite(total):raise ValueError('nonfinite cut classifier objective')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                metrics,logits=evaluate(model,x,c,dev);record=dict(epoch=epoch,metrics=metrics);history.append(record);path=directory/f'epoch{epoch:02d}.pt';torch.save(dict(record,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),path);key=(metrics['balanced_logistic'],epoch)
                if best is None or key<best:best=key;selected=path;selected_record=record
            shutil.copy2(selected,directory/'selected.pt');model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['model']);fit_metrics,_=evaluate(model,x,c,fit)
            with torch.no_grad():all_logits=model(x)
            torch.save(dict(logits=all_logits,source_ids=[s['id'] for s in c['sources']],roles=[s['role'] for s in c['sources']],checkpoint_sha256=sha(directory/'selected.pt')),directory/'selected-logits.pt');write(directory/'history.json',history);reports.append(dict(arm=arm,seed=seed,selected_epoch=selected_record['epoch'],metrics=selected_record['metrics'],fit_metrics=fit_metrics,parameters=sum(v.numel() for v in model.parameters()),checkpoint_sha256=sha(directory/'selected.pt')));write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    progress(pid=None,stage='complete',completed=15);print(json.dumps(reports))


if __name__=='__main__':main()
