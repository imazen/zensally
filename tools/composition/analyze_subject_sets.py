#!/usr/bin/env python3
"""Family-paired weak-reference analysis; retain old replay attribution controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from artifact_identity import sha

def records(directory):
    rows=json.loads((directory/'summary.json').read_text())
    for row in rows:
        path=directory/(row['arm']+'-seed'+str(row['seed']))/'best.pt'
        if sha(path)!=row['weights_sha256'] or not row['replay_exact']:raise ValueError('selected replay/weight identity differs')
    return rows

def vectors(directory,arm):
    output=[]
    for seed in [31,137,911]:
        rows=json.loads((directory/(arm+'-seed'+str(seed))/'best-predictions.json').read_text())['new'];groups=defaultdict(list)
        for row in rows:
            if row['essential']:groups[row['family_id']].append(row)
        output.append({family:np.array([np.mean([r[key] for r in group]) for key in ['covered','reference_match','explicit_face_cut']]) for family,group in groups.items()})
    families=sorted(output[0])
    if any(set(v)!=set(families) for v in output):raise ValueError('seed development families differ')
    return families,np.mean([[v[f] for f in families] for v in output],axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['fits','replay_control','cache','output']:p.add_argument(key,type=Path)
    a=p.parse_args();main=records(a.fits);control=records(a.replay_control)
    if len(main)!=12 or len(control)!=3:raise ValueError('all15 controlled fits required')
    protocol=json.loads((a.fits/'protocol.json').read_text())
    if any(r['epoch']!=0 for r in main if r['arm']=='trainable-image'):raise ValueError('this analysis baseline must be the unchanged original initializer')
    families,baseline=vectors(a.fits,'trainable-image');values={}
    for arm in ['frozen-image','trainable-image','geometry-only','permuted-image']:
        names,matrix=vectors(a.fits,arm)
        if names!=families:raise ValueError('comparison family set differs')
        values[arm]=matrix
    names,matrix=vectors(a.replay_control,'frozen-image')
    if names!=families:raise ValueError('replay-control family set differs')
    values['old-replay-only']=matrix
    rng=np.random.default_rng(31);indices=rng.integers(0,len(families),size=(5000,len(families)))
    paired={}
    for label,reference in [('initializer',baseline),('old-replay-only',values['old-replay-only']),('geometry-only',values['geometry-only']),('permuted-image',values['permuted-image'])]:
        delta=values['frozen-image']-reference;sample=delta[indices].mean(1)
        paired[label]={key:dict(delta=float(delta[:,j].mean()),ci95=[float(v) for v in np.quantile(sample[:,j],[.025,.975])]) for j,key in enumerate(['coverage','reference_match','explicit_face_cut'])}
    averages={}
    for arm,matrix in values.items():
        rows=control if arm=='old-replay-only' else [r for r in main if r['arm']==arm]
        averages[arm]=dict(**{key:float(matrix[:,j].mean()) for j,key in enumerate(['coverage','reference_match','explicit_face_cut'])},
            old_essential_regret=float(np.mean([r['old']['essential']['teacher_regret'] for r in rows])),
            old_nonregression_fits=sum(r['old_nonregression'] for r in rows),fits=len(rows))
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(models=main,replay_controls=control,averages=averages,paired=paired,new_development_families=len(families),
        registered_new_development_sources=24,all_selected_native_replays_exact=True,bootstrap=dict(replicates=5000,seed=31,unit='source family',seeds='mean of3 selected models before family bootstrap'),
        cache=json.loads((a.cache/'summary.json').read_text()),baseline_old=protocol['baseline_old'],baseline_new=protocol['baseline_new'],
        model_promoted=False,complete_face_coverage_certified=False,artistic_gold=False,
        limitation='Only finite confidently proposed, geometrically consistent windows are positives; remaining artistic compositions are unknown. This panel is development, not an untouched test.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['averages','paired','new_development_families','model_promoted']}))

if __name__=='__main__':main()

