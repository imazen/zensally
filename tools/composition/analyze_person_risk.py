#!/usr/bin/env python3
"""Family-paired selected-policy diagnostics and native pair extent distributions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from artifact_identity import sha
from distill_attention_linear import write


def bootstrap(values):
    a=np.asarray(values,float);rng=np.random.default_rng(31);draw=rng.integers(0,len(a),(2000,len(a)));return dict(families=len(a),mean=float(a.mean()),paired_family_bootstrap_95_percentile=np.quantile(a[draw].mean(-1),[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','cache','coverage','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3' or sha(a.coverage/'SHA256SUMS.json')!='160d693d91bae8dd858036c0bd198088af26be847122e50bdad8fa2ec336a34a':raise ValueError('frozen diagnostic inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);index={s['id']:i for i,s in enumerate(c['sources'])};meta={(r['id'],r['query']):r for r in json.loads((a.coverage/'queries.json').read_text())};heads={r['id']:r['head_natives'] for r in json.loads((a.coverage/'sources.json').read_text())};pred=[json.loads(r) for r in (a.study/'predictions.jsonl').read_text().splitlines()];selected=json.loads((a.study/'selected.json').read_text());rows=[r for r in pred if r['kind']=='soft' and (r['applicability']=='person' or r['model']=='unfiltered')];lookup={(r['model'],r['id'],r['query']):r for r in rows};keys=sorted((r['id'],r['query']) for r in rows if r['model']=='unfiltered' and r['query']<18 and meta[r['id'],r['query']]['reference_known'] and meta[r['id'],r['query']]['head_known']);groups={}
    for key in keys:groups.setdefault(meta[key]['family_id'],[]).append(key)
    def native(row):
        i=index[row['id']];w,h=c['sources'][i]['native_size'];return (c['query_boxes'][i,row['query'],row['selected']]*torch.tensor([w,h,w,h])).round().int().tolist()
    def retained(row):
        b=native(row);return any(b[0]<=h[0] and b[1]<=h[1] and b[2]>=h[2] and b[3]>=h[3] for h in heads[row['id']])
    comparisons=[]
    for control in ['unfiltered','permuted-attention']:
        for metric in ['weak_cut','reference_match','head_retained']:
            delta=[]
            for family,requests in groups.items():
                per_seed=[]
                for seed in [31,137,911]:
                    one=[]
                    for sid,q in requests:
                        real=lookup[f'hidden-attention-seed{seed}',sid,q];other=lookup['unfiltered' if control=='unfiltered' else f'{control}-seed{seed}',sid,q]
                        if not real['offered'] or not other['offered']:raise ValueError('soft full-offer diagnostic lost coverage')
                        one.append(int(retained(real))-int(retained(other)) if metric=='head_retained' else int(real[metric])-int(other[metric]))
                    per_seed.append(sum(one)/len(one))
                delta.append(sum(per_seed)/3)
            comparisons.append(dict(real='hidden-attention',control=control,metric=metric,requests=len(keys),**bootstrap(delta)))
    extents=[]
    for name in ['unfiltered',*[f'hidden-attention-seed{s}' for s in [31,137,911]]]:
        ratios=[];changed=0
        for source in [s for s in c['sources'] if s['role']=='dev']:
            for ri in range(11):
                pair=[lookup[name,source['id'],ri*2+m] for m in range(2)];outer,inner=map(native,pair);oa=(outer[2]-outer[0])*(outer[3]-outer[1]);ia=(inner[2]-inner[0])*(inner[3]-inner[1])
                if not 0<ia<oa:raise ValueError('strictly smaller native pair required')
                ratios.append(ia/oa);changed+=any(pair[m]['selected']!=lookup['unfiltered',source['id'],ri*2+m]['selected'] for m in range(2))
        extents.append(dict(model=name,requests=len(ratios),changed_pairs=changed,inner_outer_area_ratio_min=float(min(ratios)),median=float(np.median(ratios)),p95=float(np.quantile(ratios,.95)),maximum=float(max(ratios))))
    a.output.mkdir(exist_ok=False);result=dict(comparisons=comparisons,extents=extents,selected_soft_original_passes={mode:sum(r['old']['original_full_gate'] for r in selected if r['applicability']==mode) for mode in ['all','person']},selected_soft_nonzero_strengths=sum(r['strength']>0 for r in selected),bootstrap=dict(repeats=2000,seed=31,seeds_averaged_before_family_resampling=True),limitation='same reused DEV panel; scalar diagnostic estimates are not independent TEST confirmation, excellent crop labels or complete face guarantees',api_calls=0,test_inference=False,model_promoted=False);write(a.output/'summary.json',result);print(json.dumps(result))


if __name__=='__main__':main()
