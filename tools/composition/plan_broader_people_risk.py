#!/usr/bin/env python3
"""Freeze count/scale-stratified DEV families outside previous risk/review panels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from artifact_identity import sha
from audit_face_corroboration import verify_member

CELLS=['individual-1/small','individual-1/large','individual-2/small','individual-2/large','individual-3-5/small','individual-3-5/large','individual-6+/small','group-only/unresolved']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','review','rank_review','risk','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    fixed=[(a.acquired/'SHA256SUMS.json','93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1'),(a.review/'SHA256SUMS.json','75a75ee2df2c85ecac6b18cf6c875d3bfb0c6d625bf26542efc57fcbb4d2bca9'),(a.review/'source-mask.json','e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80'),(a.risk/'SHA256SUMS.json','0fbcd218ec4012a72113ab809fa8c489455a9182a65c1f54caf559c9f8968080')]
    if any(sha(path)!=digest for path,digest in fixed):raise ValueError('frozen people inputs differ')
    for root,name in [(a.acquired,'manifest.json'),(a.review,'source-mask.json'),(a.risk,'fits/proposals.json')]:verify_member(root,name)
    previous=json.loads(a.rank_review.read_text())['sources']
    if len(previous)!=8:raise ValueError('all prior eight reviewed families required')
    recent=json.loads((a.risk/'fits/proposals.json').read_text())['cohorts']['new']['sources']
    if len(recent)!=72:raise ValueError('recent source panel differs')
    excluded={s['family_id'] for s in [*previous,*recent]};excluded_ids={s['id'] for s in [*previous,*recent]};mask={s['id']:s for s in json.loads((a.review/'source-mask.json').read_text())};rows=json.loads((a.acquired/'manifest.json').read_text())['rows'];selected=[];pool=[]
    for row in rows:
        s=row['source'];m=mask[row['id']]
        if m['role']!=s['role'] or m['family_id']!=s['family_id']:raise ValueError('frozen roles differ')
        if s['role']=='dev' and m['person_auxiliary_source_eligible'] and s['family_id'] not in excluded:
            if s['id'] in excluded_ids:raise ValueError('excluded source alias changed family')
            pool.append(row)
    for cell in CELLS:
        candidates=sorted([r for r in pool if r['source']['group_cell']==cell],key=lambda r:hashlib.sha256(('zensally-broader-people-risk-v1\0'+r['id']).encode()).digest())
        if len(candidates)<2:raise ValueError('two-source cell quota unavailable: '+cell)
        for row in candidates[:2]:
            source=row['source'];key=row['id'].split(':')[1]
            selected.append(dict(id=row['id'],role='dev',family_id=source['family_id'],group_cell=cell,native_size=row['header']['size'],working_size=[row['geometry']['width'],row['geometry']['height']],rgb=str(a.acquired/key/'preview.rgb'),rgb_sha256=row['preview.rgb_sha256'],png=str(a.acquired/key/'preview.png'),png_sha256=row['preview.png_sha256'],source=source['source_information']['source']))
    if len(selected)!=16 or len({s['family_id'] for s in selected})!=16 or any(s['family_id'] in excluded for s in selected):raise ValueError('fixed sixteen disjoint families required')
    result=dict(schema='broader-people-risk-plan-v1',sources=selected,inputs={**{str(p):v for p,v in fixed},str(a.rank_review):sha(a.rank_review)},eligible_dev_pool=len(pool),selected_cells=dict(Counter(s['group_cell'] for s in selected)),excluded_recent_families=len(excluded),source_roles_changed=False,training_labels_admitted=False,test_pixels_opened=False,api_calls=0,prior_exposure='source-preview and broad local rank DEV; not recent risk FIT or eight reviewed crop families')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(sources=16,eligible_dev_pool=len(pool),selected_cells=result['selected_cells'],plan_sha256=sha(a.output))))


if __name__=='__main__':main()
