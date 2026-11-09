#!/usr/bin/env python3
"""Seed-averaged paired family bootstrap of registered attention diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    progress=json.loads((a.root/'PROGRESS.json').read_text())
    if progress['stage']!='complete' or progress['completed']!=15 or progress['pid'] is not None:raise ValueError('complete15-fit panel required')
    summary=json.loads((a.root/'summary.json').read_text());groups=defaultdict(lambda:defaultdict(list))
    for fit in summary['fits']:
        checkpoint=torch.load(a.root/(fit['arm']+'-seed'+str(fit['seed']))/'selected.pt',map_location='cpu',weights_only=False)
        families=defaultdict(list)
        for row in checkpoint['rows']:
            if row['essential']:families[row['family_id']].append(row)
        for family,rows in families.items():groups[fit['arm']][family].append([sum(r[k] for r in rows)/len(rows) for k in ['cut','match']])
    baseline=groups['baseline'];families=sorted(baseline);rows=[]
    rng=np.random.default_rng(31);draw=rng.integers(0,len(families),(2000,len(families)))
    base=np.array([baseline[f] for f in families]).mean(1)
    for arm,values in groups.items():
        if sorted(values)!=families or any(len(values[f])!=3 for f in families):raise ValueError('matched families and three seeds required')
        means=np.array([values[f] for f in families]).mean(1);delta=means-base;boot=delta[draw].mean(1)
        rows.append(dict(arm=arm,families=len(families),seeds=3,**{key:dict(mean=float(means[:,j].mean()),difference_from_baseline=float(delta[:,j].mean()),paired_family_bootstrap_95_percentile=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,key in enumerate(['head_cut','reference_match'])}))
    result=dict(rows=rows,bootstrap_repeats=2000,bootstrap_seed=31,seeds_averaged_before_resampling=True,selection_uses_same_dev_labels=True,
        api_calls=0,model_promoted=False,old_cohort_gate_evaluated=False,
        limitation='Small reused development cohort and weak references; selection-biased descriptive intervals, not independent artistic quality or complete face safety.')
    with (a.root/'analysis.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
