#!/usr/bin/env python3
"""Paired family diagnostics with fixed common-offer support across three seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from artifact_identity import sha
from candidate_cut_head import balanced_source_loss
from distill_attention_linear import write


def bootstrap(values):
    a=np.asarray(values,float);rng=np.random.default_rng(31);draw=rng.integers(0,len(a),size=(2000,len(a)));return dict(families=len(a),mean=float(a.mean()),paired_family_bootstrap_95_percentile=np.quantile(a[draw].mean(-1),[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('registered cache differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);ix=[i for i,s in enumerate(c['sources']) if s['role']=='dev'];scores={};families={}
    for row in json.loads((a.fits/'summary.json').read_text())['fits']:
        name=f"{row['arm']}-seed{row['seed']}";saved=torch.load(a.fits/name/'selected-logits.pt',map_location='cpu',weights_only=False)
        if saved['checkpoint_sha256']!=row['checkpoint_sha256'] or saved['source_ids']!=[s['id'] for s in c['sources']]:raise ValueError('frozen selected logit identity differs')
        loss,valid=balanced_source_loss(saved['logits'][ix],c['positive'][ix],c['negative'][ix]);scores[name]=loss
        for j,i in enumerate(ix):
            if valid[j]:families.setdefault(c['sources'][i]['family_id'],[]).append(j)
    families={k:sorted(set(v)) for k,v in families.items()};comparisons=[]
    for real,control in [('hidden-attention','permuted-attention'),('hidden-attention','hidden'),('hidden-attention','geometry'),('hidden-pose','hidden')]:
        difference=torch.stack([scores[f'{real}-seed{s}']-scores[f'{control}-seed{s}'] for s in [31,137,911]]).mean(0);comparisons.append(dict(real=real,control=control,metric='known-class balanced logistic; real minus control',**bootstrap([float(difference[indices].mean()) for indices in families.values()])))
    rows=json.loads((a.fits/'routing/predictions.json').read_text());lookup={(r['model'],r['id'],r['query']):r for r in rows if r['policy']=='corrected' and r['essential'] and r['head_known'] and r['reference_known']};models=[f'{arm}-seed{s}' for arm in ['hidden-attention','permuted-attention'] for s in [31,137,911]];requests={(r['id'],r['query']) for r in lookup.values()};common=sorted(key for key in requests if all(lookup[model,*key]['offered'] for model in models));grouped={}
    for key in common:grouped.setdefault(lookup[models[0],*key]['family_id'],[]).append(key)
    deltas=[];per_seed=[]
    for family,keys in grouped.items():deltas.append(sum(sum(int(lookup[f'hidden-attention-seed{s}',*k]['weak_cut'])-int(lookup[f'permuted-attention-seed{s}',*k]['weak_cut']) for k in keys)/len(keys) for s in [31,137,911])/3)
    for seed in [31,137,911]:
        r=[lookup[f'hidden-attention-seed{seed}',*k] for k in common];p=[lookup[f'permuted-attention-seed{seed}',*k] for k in common];per_seed.append(dict(seed=seed,requests=len(common),aligned_cuts=sum(v['weak_cut'] for v in r),permuted_cuts=sum(v['weak_cut'] for v in p),repairs=sum(v['weak_cut'] and not u['weak_cut'] for u,v in zip(r,p)),introduced=sum(u['weak_cut'] and not v['weak_cut'] for u,v in zip(r,p))))
    a.output.mkdir(exist_ok=False);result=dict(classification=comparisons,common_offer=dict(requests=len(common),**bootstrap(deltas),per_seed=per_seed),bootstrap=dict(repeats=2000,seed=31,seeds_averaged_before_family_resampling=True),limitation='same repeatedly consulted DEV labels select classifier epochs; candidate slots are correlated; common-offer results are conditional, not independent TEST evidence or full-offer safety',api_calls=0,test_inference=False,model_promoted=False);write(a.output/'summary.json',result);print(json.dumps(result))


if __name__=='__main__':main()
