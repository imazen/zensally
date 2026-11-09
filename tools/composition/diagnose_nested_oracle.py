#!/usr/bin/env python3
"""Separate incomplete automatic head proposals from exact interval feasibility."""
import argparse
from collections import Counter
import json
from pathlib import Path
from artifact_identity import sha
from audit_face_corroboration import verify_member
from refinement_inputs import reference_match,accepted_options
from refine_nested_origin import refine
from subject_policy import face_relation
from distill_attention_linear import write
from crop_corridors import audit
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['geometry','coverage','cache','references','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.geometry.parent/'SHA256SUMS.json')!='6c8dada6dc31c97770933c1dd50a00ea73586e64baae665adb3dce18bef9b165':raise ValueError('frozen geometric inputs differ')
    if sha(a.coverage/'SHA256SUMS.json')!='160d693d91bae8dd858036c0bd198088af26be847122e50bdad8fa2ec336a34a' or sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('frozen nominations/cache differ')
    for name in ['geometry/parents.jsonl','geometry/predictions.jsonl','geometry/sources.json']:verify_member(a.geometry.parent,name)
    verify_member(a.coverage,'sources.json');c=torch.load(a.cache,map_location='cpu',weights_only=False);index={s['id']:i for i,s in enumerate(c['sources'])};sources={s['id']:s for s in c['sources']};scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().int();options,heads,reference_ids,replayed=accepted_options({'data':c,'native':native},a.references);nominations={r['id']:r for r in json.loads((a.coverage/'sources.json').read_text())}
    if any(heads[sid]!=r['head_natives'] for sid,r in nominations.items()):raise ValueError('independent nominated-head identity replay differs')
    parents=[json.loads(line) for line in (a.geometry/'parents.jsonl').read_text().splitlines()];automatic=[json.loads(line) for line in (a.geometry/'predictions.jsonl').read_text().splitlines()];automatic={(r['seed'],r['id'],r['ratio_index']):r for r in automatic if r['cohort']=='new' and r['policy']=='faces-pose/pad25'};records=[]
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(build_inputs={str(path):sha(path) for path in [a.geometry.parent/'SHA256SUMS.json',a.coverage/'SHA256SUMS.json',a.cache]},reference_inputs=reference_ids,positive_mask_replayed_queries=replayed,oracle='all existing valid nominated heads, zero additional padding; exact same refinement; diagnostic only',training_eligible=False,api_calls=0,test_inference=False,model_promoted=False))
    for parent in parents:
        if parent['cohort']!='new':continue
        sid=parent['id'];s=sources[sid];i=index[sid];ri=parent['ratio_index'];w,h=s['native_size'];known=s['annotation_valid'] and bool(heads[sid]);provided=heads[sid] if known else [];result=refine(w,h,parent['outer'],parent['inner'],parent['ratio'],provided);row=dict(seed=parent['seed'],id=sid,role=s['role'],family_id=s['family_id'],ratio=parent['ratio'],ratio_index=ri,head_known=known,**result);auto=automatic[parent['seed'],sid,ri]
        for prefix,pair in [('parent',parent),('automatic',auto),('oracle',result)]:
            row[prefix]=[dict(offered=pair[key] is not None,cut=any(face_relation(pair[key],v)=='cut' for v in heads[sid]) if pair[key] is not None else False,head_retained=any(face_relation(pair[key],v)=='contained' for v in heads[sid]) if pair[key] is not None else False,match=reference_match(pair[key],options.get((sid,ri*2+m),[]),w,h)) for m,key in enumerate(['outer','inner'])]
        row['reference_known']=[bool(c['reference_valid'][i,ri*2+m].any()) for m in range(2)];row['interval_safe']=audit(result['inner'],result['outer'],w,h,parent['ratio'],heads[sid])['geometrically_admissible'] if result['inner'] is not None and known else None
        if known and (any(v['cut'] for v in row['oracle']) or (all(v['offered'] for v in row['oracle']) and row['interval_safe'] is not True)):raise ValueError('nominated-head oracle violates its own geometric constraints')
        records.append(row)
    summaries=[]
    for seed in [31,137,911]:
        for role in ['fit','dev']:
            for secondary in [False,True]:
                rows=[r for r in records if r['seed']==seed and r['role']==role and (r['ratio_index']>=9)==secondary and r['head_known']];counts={}
                for prefix in ['parent','automatic','oracle']:
                    pairs=[(r,m) for r in rows for m in range(2) if r['reference_known'][m]];counts[prefix]=dict(requests=len(pairs),offered=sum(r[prefix][m]['offered'] for r,m in pairs),cuts=sum(r[prefix][m]['cut'] for r,m in pairs),matches=sum(r[prefix][m]['match'] for r,m in pairs),head_retained=sum(r[prefix][m]['head_retained'] for r,m in pairs))
                pairs=[(r,m) for r in rows for m in range(2) if r['reference_known'][m] and r['oracle'][m]['offered']];summaries.append(dict(seed=seed,role=role,secondary=secondary,known_pairs=len(rows),oracle_safe_intervals=sum(r['interval_safe'] is True for r in rows),statuses=dict(Counter(r['status'] for r in rows)),common_offer_parent_cuts=sum(r['parent'][m]['cut'] for r,m in pairs),common_offer_automatic_cuts=sum(r['automatic'][m]['cut'] and r['automatic'][m]['offered'] for r,m in pairs),**counts))
    write(a.output/'summary.json',dict(rows=summaries,api_calls=0,test_inference=False,model_promoted=False))
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps([r for r in summaries if r['role']=='dev']))


if __name__=='__main__':main()
