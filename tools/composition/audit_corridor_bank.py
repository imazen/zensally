#!/usr/bin/env python3
"""Measure fixed candidate-bank support without treating uncovered crops as bad."""
import argparse
from collections import defaultdict
import json
from statistics import median
from pathlib import Path
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from crop_corridors import member,nearest_crop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','pilot','analysis','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed C002/hybrid candidate cache differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(cache['sources'])}
    report=json.loads((a.analysis/'summary.json').read_text());cases={r['case_id']:r for r in json.loads((a.pilot/'plan.json').read_text())['cases']};group_sources={}
    for path in a.pilot.rglob('plan.json'):
        for group in json.loads(path.read_text()).get('groups',[]):group_sources[group['case_id']]=group['source_id']
    rows=[]
    for scope,intervals in [('strict-excellent',[r for r in report['intervals'] if r['eligible_for_weak_interval_admission']]),
        ('consistent-usable-diagnostic',[r for r in report['intervals'] if r['pair'].get('pair_protocol',1)==2 and r['complete_three_presentations'] and r['all_reviewed_samples_accepted']])]:
        groups=defaultdict(list)
        for interval in intervals:groups[(group_sources[interval['group']],interval['ratio'])].append(interval)
        for (sid,ratio),values in groups.items():
            i=lookup[sid];source=cache['sources'][i];w,h=source['native_size'];scale=torch.tensor([w,h,w,h])
            if any(r['family_id']!=source['family_id'] or r['role']!=source['role'] for r in values):raise ValueError('fixed source role/family differs')
            for mode in ['minimal','maximal']:
                q=cache['ratios'].index(ratio)*2+(mode=='maximal');bank=(cache['query_boxes'][i,q]*scale).round().int().tolist();eligible=cache['eligible'][i,q].tolist()
                inside=[bool(ok and any(member(box,r['pair']['inner'],r['pair']['outer'],ratio) for r in values)) for box,ok in zip(bank,eligible)]
                nearest=[min((nearest_crop(box,r['pair']['inner'],r['pair']['outer'],ratio,w,h,minimal=mode=='minimal') for r in values),key=lambda p:(p[1],p[0])) if ok else None for box,ok in zip(bank,eligible)]
                choice=int(cache['initial_scores'][i,q].masked_fill(~cache['eligible'][i,q],-torch.inf).argmax())
                rows.append(dict(scope=scope,id=sid,role=source['role'],family_id=source['family_id'],ratio=ratio,mode=mode,eligible_candidates=sum(eligible),interval_candidates=sum(inside),
                    covered=any(inside),selected=choice,selected_inside=inside[choice],selected_native_box=bank[choice],old_reference_available=bool(cache['reference_valid'][i,q].any()),
                    nearest_bank_distance_128_grid=min(r[1] for r in nearest if r is not None),selected_distance_128_grid=nearest[choice][1],nearest_selected_interval_box=nearest[choice][0],
                    old_selected_reference_match=bool(cache['positive'][i,q,choice]) if bool(cache['reference_valid'][i,q].any()) else None,
                    selected_known_head_cut=bool(cache['negative'][i,q,choice]),outside_interval_is_unknown_not_a_negative=True))
    summary=[]
    for scope in ['strict-excellent','consistent-usable-diagnostic']:
        for mode in ['minimal','maximal']:
            selected=[r for r in rows if r['scope']==scope and r['mode']==mode]
            summary.append(dict(scope=scope,mode=mode,requests=len(selected),families=len({r['family_id'] for r in selected}),covered=sum(r['covered'] for r in selected),
                fixed_C002_inside=sum(r['selected_inside'] for r in selected),interval_candidates=sum(r['interval_candidates'] for r in selected),
                nearest_bank_distance_128_grid=dict(min=min(r['nearest_bank_distance_128_grid'] for r in selected),median=median(r['nearest_bank_distance_128_grid'] for r in selected),max=max(r['nearest_bank_distance_128_grid'] for r in selected)) if selected else None,
                selected_distance_128_grid=dict(min=min(r['selected_distance_128_grid'] for r in selected),median=median(r['selected_distance_128_grid'] for r in selected),max=max(r['selected_distance_128_grid'] for r in selected)) if selected else None))
    a.output.mkdir(exist_ok=False);result=dict(summary=summary,rows=rows,trained=False,api_calls=0,model_promoted=False,test_inference=False,
        fixed_cache_sha256=sha(a.cache),pilot_analysis_sha256=sha(a.analysis/'summary.json'),
        limitation='Exact interval support, not artistic grades for arbitrary model outputs. No excellent intervals are currently admitted; usable ranges are a separate diagnostic. Outside is unknown, not a failed-quality label.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
