#!/usr/bin/env python3
"""Matched family comparisons with independently replayed original quality gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits',type=Path);p.add_argument('gates',type=Path);p.add_argument('output',type=Path);a=p.parse_args();torch.set_num_threads(2)
    fits=json.loads((a.fits/'summary.json').read_text())['fits'];gates=json.loads((a.gates/'summary.json').read_text())
    if len(fits)!=12 or len(gates['rows'])!=12:raise ValueError('complete matched12 panels required')
    vectors={};keys=None;epochs={};gate_lookup={(r['arm'],r['seed']):r for r in gates['rows']}
    for fit in fits:
        name=fit['arm']+'-seed'+str(fit['seed']);state=torch.load(a.fits/name/'selected.pt',map_location='cpu',weights_only=False)
        rows=[r for r in state['rows'] if r['essential']];current={(r['id'],r['family_id'],r['query']) for r in rows}
        if len(current)!=len(rows) or keys is not None and current!=keys:raise ValueError('matched unique essential queries required')
        keys=current;groups=defaultdict(list)
        for row in rows:groups[row['family_id']].append(row)
        vectors[(fit['arm'],fit['seed'])]={f:np.array([np.mean([r[k] for r in group]) for k in ['cut','match']]) for f,group in groups.items()};epochs[name]=state['epoch']
    families=sorted(groups);rng=np.random.default_rng(31);draw=rng.integers(0,len(families),(2000,len(families)));comparisons=[];averages=[]
    for arm in ['replay-only','replay-contrastive','real','permuted-target']:
        matrix=np.array([[vectors[(arm,seed)][f] for f in families] for seed in [31,137,911]])
        averages.append(dict(arm=arm,head_cut=float(matrix[:,:,0].mean()),reference_match=float(matrix[:,:,1].mean()),full_old_gate_passes=sum(gate_lookup[(arm,seed)]['old_essential_nonregression'] for seed in [31,137,911]),fits=3,old_regrets=[gate_lookup[(arm,seed)]['metrics']['essential']['teacher_regret'] for seed in [31,137,911]]))
    for left,right in [('real','permuted-target'),('real','replay-contrastive'),('replay-contrastive','replay-only')]:
        delta=np.array([[vectors[(left,seed)][f]-vectors[(right,seed)][f] for f in families] for seed in [31,137,911]]);mean=delta.mean(0);boot=mean[draw].mean(1)
        comparisons.append(dict(left=left,right=right,**{k:dict(mean=float(mean[:,j].mean()),seed_differences=delta[:,:,j].mean(1).tolist(),paired_family_95_percentile=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,k in enumerate(['head_cut','reference_match'])}))
    a.output.mkdir(parents=True,exist_ok=False);result=dict(averages=averages,comparisons=comparisons,selected_epochs=epochs,families=len(families),essential_queries=len(keys),bootstrap_repeats=2000,bootstrap_seed=31,seeds_averaged_before_resampling=True,api_calls=0,test_inference=False,model_promoted=False,limitation='Selected-DEV weak-label comparisons, not independent artistic quality or face-safety proof.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
