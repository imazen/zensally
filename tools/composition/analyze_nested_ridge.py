#!/usr/bin/env python3
"""Family-paired uncertainty for low-capacity nested controls on reused DEV."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from analyze_person_risk import bootstrap
from distill_attention_linear import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','cache','output']:p.add_argument(name,type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--models',nargs='+',default=['image-pca8','attention-pca8'])
    a=p.parse_args();torch.set_num_threads(1)
    if sha(a.cache)!='3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4':raise ValueError('fixed paired cache differs')
    if a.baseline is not None and sha(a.baseline/'SHA256SUMS.json')!='318f88d1d3259433ead7d424f082a5ec63cb9908d1578e242968449286f910d5':raise ValueError('frozen geometry baseline differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);models=['geometry',*a.models];rows={name:{(r['id'],r['query']):r for r in json.loads(((a.baseline if name=='geometry' and a.baseline is not None else a.study)/(name+'-predictions.json')).read_text())} for name in models};baseline=rows['geometry'];keys=[key for key,r in baseline.items() if r['role']=='dev' and r['query']<18 and r['head_known'] and r['reference_known']];families={}
    for key in keys:families.setdefault(baseline[key]['family_id'],[]).append(key)
    if len(keys)!=139 or len(families)!=17:raise ValueError('fixed accepted DEV panel differs')
    comparisons=[]
    for arm in models[1:]:
        for metric in ['reference_match','weak_cut','head_retained']:
            values=[sum(int(rows[arm][key][metric])-int(baseline[key][metric]) for key in members)/len(members) for members in families.values()];comparisons.append(dict(arm=arm,control='geometry',metric=metric,requests=len(keys),**bootstrap(values)))
        values=[]
        for i,s in enumerate(c['sources']):
            if s['role']!='dev':continue
            errors=[]
            for ri in range(9):
                if not c['target_valid'][i,ri].any():continue
                scale=torch.tensor([*s['native_size'],*s['native_size']],dtype=torch.float64);target=c['targets'][i,ri,0];one=[]
                for name in [arm,'geometry']:
                    pair=torch.tensor([rows[name][s['id'],ri*2+m]['native'] for m in range(2)],dtype=torch.float64)/scale;one.append(float(((pair-target).abs()*128).max()))
                errors.append(one[0]-one[1])
            if errors:values.append(sum(errors)/len(errors))
        comparisons.append(dict(arm=arm,control='geometry',metric='paired_max_edge_error',requests=36,**bootstrap(values)))
    result=dict(comparisons=comparisons,native_changed_from_geometry={arm:sum(r['native']!=baseline[key]['native'] for key,r in rows[arm].items()) for arm in models[1:]},inputs={str(path):sha(path) for path in [a.cache,a.study/'summary.json',*((a.baseline/'SHA256SUMS.json',) if a.baseline is not None else ())]},bootstrap=dict(repeats=2000,seed=31,family_paired=True),limitation='same repeatedly used DEV panel; descriptive uncertainty, not independent TEST or artistic-quality confirmation',api_calls=0,test_inference=False,model_promoted=False);write(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
