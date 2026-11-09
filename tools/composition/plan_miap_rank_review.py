#!/usr/bin/env python3
"""Fix eight varied people review sources independently of crop-model scores."""
import argparse
import hashlib
import json
from pathlib import Path

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','review','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.acquired/'SHA256SUMS.json')!='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1' or sha(a.review/'source-mask.json')!='e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80':
        raise ValueError('frozen photographic review inputs differ')
    mask={s['id']:s for s in json.loads((a.review/'source-mask.json').read_text())}
    rows=json.loads((a.acquired/'manifest.json').read_text())['rows'];selected=[]
    cells=['individual-1/small','individual-1/large','individual-2/small','individual-2/large',
        'individual-3-5/small','individual-3-5/large','individual-6+/small','group-only/unresolved']
    for cell in cells:
        choices=[r for r in rows if r['source']['group_cell']==cell and r['source']['role']=='dev' and mask[r['id']]['person_auxiliary_source_eligible']]
        if not choices:raise ValueError('fixed review cell empty')
        source=min(choices,key=lambda r:hashlib.sha256(('miap-rank-review-v1\0'+r['id']).encode()).digest())
        selected.append(dict(id=source['id'],family_id=source['source']['family_id'],role='dev',group_cell=cell))
    plan=dict(sources=selected,selection='eight registered count/scale cells, minimum fixed source hash',
        chosen_without_detector_or_crop_quality_predictions=True,ratios=['3:1','1:3','1:1','6:1','1:6'],modes=['minimal','maximal'],
        screenshot_budget_bytes=96*1024**2,human_artistic_labels=False,face_coverage_certified=False,paid_model_api_calls=0)
    with a.output.open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(sources=len(selected),plan_sha256=sha(a.output))))


if __name__=='__main__':main()
