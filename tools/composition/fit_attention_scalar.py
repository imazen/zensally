#!/usr/bin/env python3
"""Registered frozen-encoder attention ablations on cached weak references."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from attention_scalar_head import AttentionScalarHead
from acceptable_crop_sets import acceptable_set_loss

ARMS={'baseline':0,'pose':16,'pose-gaze':24,'pose-gaze-paths':32,'permuted-full':32}


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


@torch.no_grad()
def evaluate(model,hidden,extras,cache,indices):
    scores=model(hidden[indices],extras[indices])
    if not torch.isfinite(scores).all():raise ValueError('nonfinite predictions')
    choice=scores.masked_fill(~cache['eligible'][indices],-torch.inf).argmax(-1)
    valid=cache['reference_valid'][indices].any(-1)
    cut=cache['negative'][indices].gather(-1,choice[...,None])[...,0]
    match=cache['positive'][indices].gather(-1,choice[...,None])[...,0]
    rows=[]
    for j,i in enumerate(indices):
        for q in range(22):
            if bool(valid[j,q]):rows.append(dict(id=cache['sources'][i]['id'],family_id=cache['sources'][i]['family_id'],query=q,essential=q<18,selected=int(choice[j,q]),cut=bool(cut[j,q]),match=bool(match[j,q])))
    metrics={}
    for scope in ['all','essential']:
        groups={}
        for row in rows:
            if scope=='essential' and not row['essential']:continue
            groups.setdefault(row['family_id'],[]).append(row)
        if not groups:raise ValueError('empty evaluation cohort')
        metrics[scope]=dict(requests=sum(map(len,groups.values())),families=len(groups),**{key:sum(sum(r[key] for r in group)/len(group) for group in groups.values())/len(groups) for key in ['cut','match']})
    return metrics,rows,scores.cpu()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    identity=sha(a.cache)
    if identity!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered cache differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    fit=[i for i,s in enumerate(cache['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(cache['sources']) if s['role']=='dev']
    if len(fit)!=48 or len(dev)!=24 or {cache['sources'][i]['family_id'] for i in fit}&{cache['sources'][i]['family_id'] for i in dev}:raise ValueError('fixed independent roles required')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(cache_sha256=identity,revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),arms=ARMS,seeds=[31,137,911],epochs=88,batch_sources=12,queries_per_source=22,optimizer='AdamW lr0.001 decay0.0001 clip5 FP32 CPU two threads',normalization='FIT eligible candidates per column, population std floor0.001; separate donor normalization',selection='essential family mean head-cut rate, then negative reference match, then earliest epoch including zero',api_calls=0,nested_labels_used=False,test_inference=False,model_promoted=False))
    status=dict(pid=os.getpid(),stage='fitting',completed=0,required=15)
    def progress(**fields):status.update(fields,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failed():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);reports=[];hidden=cache['hidden'];initial=cache['initial_scores']
    for arm,active in ARMS.items():
        raw=cache['permuted_extras'] if arm=='permuted-full' else cache['extras']
        train_values=raw[fit][cache['eligible'][fit]]
        mean=train_values.mean(0);std=train_values.std(0,unbiased=False).clamp_min(.001);extra=(raw-mean)/std
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=AttentionScalarHead(cache['scorer_state'],active)
            if not torch.equal(model(hidden,extra),initial):raise ValueError('initial frozen replay differs')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            directory=a.output/(arm+'-seed'+str(seed));directory.mkdir();history=[];best_key=None
            for epoch in range(89):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),12):
                        indices=order[start:start+12];optimizer.zero_grad(set_to_none=True)
                        scores=model(hidden[indices],extra[indices]);loss,valid=acceptable_set_loss(scores,cache['positive'][indices],cache['negative'][indices])
                        source_valid=valid.any(-1)
                        if not source_valid.any():raise ValueError('batch lacks contrastive supervision')
                        objective=((loss*valid).sum(-1)/valid.sum(-1).clamp_min(1))[source_valid].mean()
                        if not torch.isfinite(objective):raise ValueError('nonfinite loss')
                        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                        if not torch.equal(model.extra.weight[:,active:],torch.zeros_like(model.extra.weight[:,active:])):raise ValueError('inactive features changed')
                metrics,rows,scores=evaluate(model,hidden,extra,cache,dev);history.append(dict(epoch=epoch,metrics=metrics))
                key=(metrics['essential']['cut'],-metrics['essential']['match'],epoch)
                state=dict(epoch=epoch,model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),mean=mean,std=std,metrics=metrics,rows=rows,scores=scores)
                if epoch==0:torch.save(state,directory/'initial.pt')
                if best_key is None or key<best_key:
                    best_key=key;torch.save(state,directory/'selected.pt');selected_epoch=epoch;selected_metrics=metrics
                if epoch==88:torch.save(state,directory/'terminal.pt')
            write(directory/'history.json',history)
            reports.append(dict(arm=arm,seed=seed,selected_epoch=selected_epoch,metrics=selected_metrics))
            write(a.output/'summary.json',dict(fits=reports,api_calls=0,nested_labels_used=False,test_inference=False,model_promoted=False,old_cohort_gate_evaluated=False))
    progress(pid=None,stage='complete',completed=len(reports));print(json.dumps(reports))


if __name__=='__main__':main()
