#!/usr/bin/env python3
"""Direct matched attention controls, retaining per-seed and family uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(2);values={};keys=None;epochs={}
    summary=json.loads((a.fits/'summary.json').read_text())
    for fit in summary['fits']:
        name=fit['arm']+'-seed'+str(fit['seed']);checkpoint=torch.load(a.fits/name/'selected.pt',map_location='cpu',weights_only=False)
        rows=[r for r in checkpoint['rows'] if r['essential']];current={(r['id'],r['family_id'],r['query']) for r in rows}
        if len(current)!=len(rows) or keys is not None and current!=keys:raise ValueError('matched unique essential queries required')
        keys=current;families=defaultdict(list)
        for r in rows:families[r['family_id']].append(r)
        values[(fit['arm'],fit['seed'])]={f:np.array([np.mean([r[k] for r in rs]) for k in ['cut','match']]) for f,rs in families.items()}
        epochs[name]=checkpoint['epoch']
    families=sorted(families);rng=np.random.default_rng(31);draw=rng.integers(0,len(families),(2000,len(families)));results=[]
    for left,right in [('pose','baseline'),('pose-gaze','permuted-full'),('pose-gaze-paths','permuted-full'),('pose-gaze','pose'),('pose-gaze-paths','pose-gaze')]:
        deltas=np.array([[values[(left,seed)][f]-values[(right,seed)][f] for f in families] for seed in [31,137,911]])
        average=deltas.mean(0);boot=average[draw].mean(1)
        results.append(dict(left=left,right=right,**{key:dict(mean_difference=float(average[:,j].mean()),seed_differences=deltas[:,:,j].mean(1).tolist(),paired_family_95_percentile=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,key in enumerate(['head_cut','reference_match'])}))
    result=dict(comparisons=results,selected_epochs=epochs,essential_queries=len(keys),families=len(families),bootstrap=dict(replicates=2000,seed=31,seeds_averaged_before_family_resampling=True),
        api_calls=0,trained=False,model_promoted=False,limitation='Descriptive selected-DEV comparisons on weak labels; no independent quality or causal feature attribution claim.')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
