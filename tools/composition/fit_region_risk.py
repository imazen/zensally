#!/usr/bin/env python3
"""Matched region-relation cut-risk heads using frozen weak candidate labels."""
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
from person_risk_inputs import load
from region_risk_features import ARMS,proposal_bundle,build,features,tensor_sha
from candidate_cut_head import CutHead,balanced_source_loss
from fit_candidate_cut import evaluate
from local_rank_distillation import family_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','miap','unsplash','faces','pose','original','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    contexts,_,_,identities=load(a.parent,a.miap,a.unsplash);bundle,extra=proposal_bundle(contexts,a.faces,a.pose,a.original,a.miap,a.unsplash);identities.update(extra);c=contexts['new']['data'];fit=[i for i,s in enumerate(c['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(c['sources']) if s['role']=='dev']
    if len(fit)!=48 or len(dev)!=24 or bool((c['positive']&c['negative']).any()) or bool(((c['positive']|c['negative'])&~c['eligible']).any()):raise ValueError('fixed source/weak-label contract differs')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'proposals.json',dict(schema='region-risk-proposals-v1',cohorts=bundle,inputs=identities));write(a.output/'protocol.json',dict(inputs=identities,arms=ARMS,seeds=[31,137,911],epochs=32,batch_sources=8,normalization='FIT eligible only mean/std floor0.001',optimizer='AdamW0.001 decay0.0001 clip5 CPU FP32',selection='DEV family-balanced known-class logistic loss, earliest tie',api_calls=0,new_teacher_inference=False,test_inference=False,model_promoted=False));state=dict(pid=os.getpid(),stage='features',completed=0,required=12);reports=[]
    def progress(**kw):state.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',state)
    def failed():
        if state['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);progress();real=build(contexts['new'],bundle['new']);permuted=build(contexts['new'],bundle['new'],permuted=True);write(a.output/'feature-identities.json',dict(real=tensor_sha(real),permuted=tensor_sha(permuted),shape=list(real.shape),donors=bundle['new']['donors']));fw=family_weights(c['sources'],fit)*len(fit);lookup={i:fw[j] for j,i in enumerate(fit)}
    for arm in ARMS:
        x=features(c,real,permuted,arm);known=x[fit][c['eligible'][fit]];mean=known.mean(0);std=known.std(0,unbiased=False).clamp_min(.001);del known
        for seed in [31,137,911]:
            torch.manual_seed(seed);model=CutHead(mean,std);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);directory=a.output/f'{arm}-seed{seed}';directory.mkdir();best=None;selected=None;history=[]
            for epoch in range(33):
                progress(arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),8):
                        ix=order[start:start+8];optimizer.zero_grad(set_to_none=True);logits=model(x[ix]);loss,valid=balanced_source_loss(logits,c['positive'][ix],c['negative'][ix]);w=torch.stack([lookup[i] for i in ix])*valid
                        if not bool(w.sum()>0):raise ValueError('registered batch lacks known labels')
                        total=(loss*w).sum()/w.sum()
                        if not torch.isfinite(total):raise ValueError('nonfinite loss')
                        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                metric,_=evaluate(model,x,c,dev);record=dict(epoch=epoch,metrics=metric);history.append(record);path=directory/f'epoch{epoch:02d}.pt';torch.save(dict(record,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),path);key=(metric['balanced_logistic'],epoch)
                if best is None or key<best:best=key;selected=path;selected_record=record
            shutil.copyfile(selected,directory/'selected.pt');model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['model']);fit_metric,_=evaluate(model,x,c,fit)
            with torch.no_grad():all_logits=model(x)
            torch.save(dict(logits=all_logits,source_ids=[s['id'] for s in c['sources']],roles=[s['role'] for s in c['sources']],checkpoint_sha256=sha(directory/'selected.pt')),directory/'selected-logits.pt');write(directory/'history.json',history);report=dict(arm=arm,seed=seed,selected_epoch=selected_record['epoch'],metrics=selected_record['metrics'],fit_metrics=fit_metric,parameters=sum(p.numel() for p in model.parameters()),inputs=x.shape[-1],checkpoint_sha256=sha(directory/'selected.pt'));reports.append(report);write(a.output/'summary.json',dict(fits=reports,api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(arm=arm,seed=seed,epoch=report['selected_epoch'],dev_loss=report['metrics']['balanced_logistic'],fit_loss=fit_metric['balanced_logistic'])),flush=True)
            if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>64*1024**2:raise ValueError('64MiB evidence budget exceeded')
    progress(pid=None,stage='complete',completed=12)


if __name__=='__main__':main()
