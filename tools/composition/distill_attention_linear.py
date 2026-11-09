#!/usr/bin/env python3
"""Bounded linear feature distillation, retaining the unchanged base scorer."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import torch
from artifact_identity import sha
from fit_attention_scalar import evaluate


class Student(torch.nn.Module):
    def __init__(self,base,teacher,mean,std,nonlinear=False):
        super().__init__();self.base=torch.nn.Linear(96,1);self.base.load_state_dict(base);self.base.requires_grad_(False)
        self.probe=torch.nn.Sequential(torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Linear(32,32)) if nonlinear else torch.nn.Linear(96,32)
        last=self.probe[-1] if nonlinear else self.probe
        torch.nn.init.zeros_(last.weight);torch.nn.init.zeros_(last.bias)
        self.register_buffer('teacher',teacher.clone());self.register_buffer('mean',mean);self.register_buffer('std',std)

    def features(self,hidden):return self.probe((hidden-self.mean)/self.std)

    def forward(self,hidden,unused):return (self.base(hidden)+torch.nn.functional.linear(self.features(hidden),self.teacher))[...,0]

    def fused(self):
        if isinstance(self.probe,torch.nn.Sequential):return None
        residual=self.teacher@self.probe.weight
        weight=self.base.weight+residual/self.std
        bias=self.base.bias+(self.teacher@self.probe.bias)-(residual*self.mean/self.std).sum()
        return dict(weight=weight.detach().clone(),bias=bias.detach().clone())


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


@torch.no_grad()
def mse(model,hidden,target,eligible,indices):
    error=(model.features(hidden[indices])-target[indices]).square().mean(-1)
    mask=eligible[indices];return float(((error*mask).sum((1,2))/mask.sum((1,2))).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','teachers','output']:p.add_argument(key,type=Path)
    p.add_argument('--relationship-cache',type=Path);p.add_argument('--nonlinear',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered attention cache differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);hidden=cache['hidden'];fit=[i for i,s in enumerate(cache['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(cache['sources']) if s['role']=='dev']
    if len(fit)!=48 or len(dev)!=24:raise ValueError('fixed roles required')
    if a.relationship_cache:
        if sha(a.relationship_cache)!='916b95282b5ed329eaad9710c737314d888fd4db47e48866af4f9da6dba50d84':raise ValueError('frozen relationship cache differs')
        relationship=torch.load(a.relationship_cache,map_location='cpu',weights_only=False)
        if relationship['source_ids']!=[s['id'] for s in cache['sources']] or relationship['parent_cache_sha256']!=sha(a.cache):raise ValueError('relationship identities differ')
        pose=cache['extras'][...,:16];cache['extras']=torch.cat([pose,relationship['real']],-1);cache['permuted_extras']=torch.cat([pose,relationship['permuted']],-1)
    training=hidden[fit][cache['eligible'][fit]];mean=training.mean(0);std=training.std(0,unbiased=False).clamp_min(.001)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'architecture.json',dict(nonlinear=a.nonlinear,relationship_cache_sha256=sha(a.relationship_cache) if a.relationship_cache else None,layout='96->32 ReLU->32' if a.nonlinear else '96->32 linear',base_frozen=True))
    write(a.output/'protocol.json',dict(cache_sha256=sha(a.cache),arms=['real','permuted-target'],seeds=[31,137,911],epochs=88,batch=12,optimizer='AdamW lr0.001 decay0.0001 clip5, FP32 CPU two threads',selection='minimum DEV feature MSE including epoch zero, earliest tie',normalization='FIT eligible candidates only; hidden population std floor0.001, teacher FIT feature normalization',loss='equal source mean MSE over eligible candidates and32 teacher features',teacher='matched-seed frozen-base full residual head',api_calls=0,test_inference=False,model_promoted=False))
    status=dict(pid=os.getpid(),stage='fitting',completed=0,required=6)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failed():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);reports=[]
    for arm in ['real','permuted-target']:
        for seed in [31,137,911]:
            teacher_path=a.teachers/(('pose-relations' if a.relationship_cache else 'pose-gaze-paths')+'-seed'+str(seed))/'selected.pt';teacher=torch.load(teacher_path,map_location='cpu',weights_only=False)
            if any(not torch.equal(teacher['model']['base.'+k],v) for k,v in cache['scorer_state'].items()):raise ValueError('original teacher base differs')
            raw=cache['extras'] if arm=='real' else cache['permuted_extras'];target=(raw-teacher['mean'])/teacher['std']
            torch.manual_seed(seed);model=Student(cache['scorer_state'],teacher['model']['extra.weight'],mean,std,nonlinear=a.nonlinear)
            optimizer=torch.optim.AdamW(model.probe.parameters(),lr=.001,weight_decay=.0001);directory=a.output/(arm+'-seed'+str(seed));directory.mkdir();history=[];best=None
            for epoch in range(89):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),12):
                        ix=order[start:start+12];optimizer.zero_grad(set_to_none=True);mask=cache['eligible'][ix]
                        error=(model.features(hidden[ix])-target[ix]).square().mean(-1)
                        loss=((error*mask).sum((1,2))/mask.sum((1,2))).mean()
                        if not torch.isfinite(loss):raise ValueError('nonfinite distillation')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.probe.parameters(),5.);optimizer.step()
                value=mse(model,hidden,target,cache['eligible'],dev);metrics,rows,scores=evaluate(model,hidden,target,cache,dev)
                history.append(dict(epoch=epoch,feature_mse=value,metrics=metrics))
                state=dict(epoch=epoch,model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),feature_mse=value,metrics=metrics,rows=rows,scores=scores,fused=model.fused(),teacher_sha256=sha(teacher_path))
                if epoch==0:torch.save(state,directory/'initial.pt')
                if best is None or value<best:
                    best=value;torch.save(state,directory/'selected.pt');selected=dict(arm=arm,seed=seed,epoch=epoch,feature_mse=value,metrics=metrics)
                if epoch==88:torch.save(state,directory/'terminal.pt')
            write(directory/'history.json',history);reports.append(selected);write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False))
    progress(pid=None,stage='complete',completed=6);print(json.dumps(reports))


if __name__=='__main__':main()
