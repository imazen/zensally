#!/usr/bin/env python3
"""FIT-only pair memorization probe; no DEV selection or model promotion."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from fit_nested_pairs import paired_metrics,BASELINE
from fit_relationship_rehearsal import old_metrics
from nested_pair_training import compatibility,pair_choices,paired_top_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['hidden','rank','initializer','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    expected=[(a.hidden,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.initializer,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered inputs differ')
    old=torch.load(a.hidden,map_location='cpu',weights_only=False);rank=torch.load(a.rank,map_location='cpu',weights_only=False);s=torch.load(a.initializer,map_location='cpu',weights_only=False)['model'];model=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);model.load_state_dict(s)
    fit=[i for i,v in enumerate(old['sources']) if v['role']=='fit'];dev=[i for i,v in enumerate(old['sources']) if v['role']=='dev']
    if len(fit)!=94 or len(dev)!=25 or old['sources']!=rank['sources']:raise ValueError('fixed roles/identities differ')
    scale=torch.tensor([[*v['native_size'],*v['native_size']] for v in rank['sources']])[:,None,None,:];boxes=(rank['boxes']*scale).round().long();mask=compatibility(boxes,old['eligible'].reshape(119,11,2,128));labels=pair_choices(old['targets'].reshape(119,11,2,128),mask)
    valid=old['query_valid'].reshape(119,11,2).all(-1);requests=[(i,r) for i in fit for r in range(11) if valid[i,r]]
    if len(requests)!=1024:raise ValueError('FIT-only1024 pairs required')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},purpose='capacity diagnostic without old/new replay constraints; not a deployable fit',seed=31,epochs=8,batch_pairs=16,optimizer='AdamW lr0.003 decay0.0001 clip5 FP32 two CPU threads',loss='FIT-only strict pair cross entropy,1024 valid paired requests',early_stop='first95 percent FIT pair exact agreement at epoch boundary; otherwise eight epochs',dev_selection=False,api_calls=0,test_inference=False,model_promoted=False))
    torch.manual_seed(31);optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.003,weight_decay=.0001);history=[]
    for epoch in range(9):
        if epoch:
            order=torch.randperm(len(requests),generator=torch.Generator().manual_seed(31+epoch)).tolist()
            for start in range(0,len(order),16):
                selected=[requests[k] for k in order[start:start+16]];ix=torch.tensor([v[0] for v in selected]);ri=torch.tensor([v[1] for v in selected]);q=ri[:,None]*2+torch.arange(2);optimizer.zero_grad(set_to_none=True);scores=model(old['hidden'][ix[:,None],q],None);loss=paired_top_loss(scores,mask[ix,ri],labels[ix,ri]).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite capacity loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
        with torch.no_grad():
            scores=model(old['hidden'],None);choice=pair_choices(scores.reshape(119,11,2,128),mask);agreement=(choice[fit]==labels[fit]).all(-1)[valid[fit]];fit_regret,_=paired_metrics(scores[fit],old,mask,fit);dev_regret,_=paired_metrics(scores[dev],old,mask,dev);raw,_=old_metrics(model,old,dev)
        record=dict(epoch=epoch,fit_exact_pairs=int(agreement.sum()),fit_pair_requests=1024,fit_paired_regret=fit_regret,dev_raw_regret=raw,dev_paired_regret=dev_regret,dev_both_gates=raw<=BASELINE and dev_regret<=BASELINE);history.append(record);torch.save(dict(record,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),a.output/f'epoch{epoch:02d}.pt');write(a.output/'history.json',history)
        if float(agreement.float().mean())>=.95:break
    write(a.output/'summary.json',dict(initial=history[0],terminal=history[-1],api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(history))


if __name__=='__main__':main()
