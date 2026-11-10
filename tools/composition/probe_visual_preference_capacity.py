#!/usr/bin/env python3
"""Bounded six-comparison learnability probe, with quality evaluated afterward."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from artifact_identity import sha
from distill_attention_linear import Student,write
from fit_relationship_rehearsal import old_metrics
from fit_attention_scalar import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old','new','control','review','fits','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if sha(a.old)!='41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af' or sha(a.new)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed hidden caches differ')
    path=a.control/'relationship-rehearsal/replay-contrastive-seed31/selected.pt'
    if sha(path)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b':raise ValueError('common initializer differs')
    protocol=json.loads((a.fits/'protocol.json').read_text());labels=a.review/'weak-preferences.json'
    if protocol['inputs'][str(labels)]!=sha(labels):raise ValueError('review label identities differ')
    old=torch.load(a.old,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=Student(new['scorer_state'],state['teacher'],state['mean'],state['std'],nonlinear=True);model.load_state_dict(state)
    rows=json.loads(labels.read_text())['preferences'];lookup={s['id']:i for i,s in enumerate(new['sources'])};hidden=[];weights=[]
    if len(rows)!=6:raise ValueError('six comparisons required')
    for row in rows:
        i=lookup[row['id']]
        if new['sources'][i]['role']!='fit' or new['sources'][i]['family_id']!=row['family_id']:raise ValueError('FIT source identity differs')
        hidden.append(new['hidden'][i,row['query'],[row['preferred'],row['dispreferred']]]);weights.append(row['weight'])
    hidden=torch.stack(hidden);weights=torch.tensor(weights);reference=torch.load(a.fits/'review-seed31/epoch00.pt',map_location='cpu',weights_only=False)['review_scores']
    if not torch.equal(model(hidden,None),reference):raise ValueError('six comparison score replay differs')
    devold=[i for i,s in enumerate(old['sources']) if s['role']=='dev'];devnew=[i for i,s in enumerate(new['sources']) if s['role']=='dev'];baseline,_=old_metrics(model,old,devold);initial_metric,_,_=evaluate(model,new['hidden'],new['extras'],new,devnew)
    optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.003,weight_decay=.0001);a.output.mkdir(parents=True,exist_ok=False);history=[]
    write(a.output/'protocol.json',dict(initializer_sha256=sha(path),labels_sha256=sha(labels),max_steps=256,optimizer='AdamW lr0.003 decay0.0001 clip5 FP32 CPU one thread',objective='only weighted canonical FIT preference softplus; no quality replay',stop='first6/6 preferred scores exceed dispreferred scores, or256 steps',api_calls=0,test_inference=False,model_promoted=False))
    for step in range(257):
        if step:
            optimizer.zero_grad(set_to_none=True);scores=model(hidden,None);loss=(F.softplus(scores[:,1]-scores[:,0])*weights).sum()/weights.sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
        with torch.no_grad():
            scores=model(hidden,None);loss=(F.softplus(scores[:,1]-scores[:,0])*weights).sum()/weights.sum();correct=int((scores[:,0]>scores[:,1]).sum())
        if not torch.isfinite(loss):raise ValueError('nonfinite capacity loss')
        history.append(dict(step=step,loss=float(loss),correct=correct));torch.save(dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),review_scores=scores),a.output/f'step{step:03d}.pt')
        if correct==6:break
    regret,old_choices=old_metrics(model,old,devold);metrics,predictions,_=evaluate(model,new['hidden'],new['extras'],new,devnew)
    result=dict(steps=step,correct=correct,total=6,review_loss=float(loss),initial_old_regret=baseline,old_regret=regret,old_same_platform_initializer_nonregression=regret<=baseline,initial_metrics=initial_metric,metrics=metrics,api_calls=0,test_inference=False,model_promoted=False,scope='deliberate FIT-only overfit; no held-out artistic-quality claim')
    write(a.output/'history.json',history);write(a.output/'summary.json',result);write(a.output/'predictions.json',predictions);print(json.dumps(result))


if __name__=='__main__':main()
