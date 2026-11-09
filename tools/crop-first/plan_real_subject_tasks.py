#!/usr/bin/env python3
"""Register72 real-photo subject-choice references without changing existing roles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from subject_zones import RATIOS

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['acquired','review','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    if digest(a.acquired/'SHA256SUMS.json')!='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1':raise ValueError('acquisition packet changed')
    if digest(a.review/'source-mask.json')!='e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80':raise ValueError('review mask changed')
    rows=json.loads((a.acquired/'manifest.json').read_text())['rows']
    mask={r['id']:r for r in json.loads((a.review/'source-mask.json').read_text())}
    candidates=[r for r in rows if mask[r['id']]['person_auxiliary_source_eligible']]
    cells=sorted({r['source']['cell'] for r in candidates});selected=[]
    for role,count in [('fit',4),('dev',2)]:
        for cell in cells:
            group=[r for r in candidates if r['source']['role']==role and r['source']['cell']==cell]
            if len(group)<count:raise ValueError('registered stratified cell lacks sources')
            selected+=sorted(group,key=lambda r:hashlib.sha256(('subject-choice-v4-real|'+r['id']).encode()).hexdigest())[:count]
    if len(cells)!=12 or len(selected)!=72 or len({r['source']['family_id'] for r in selected})!=72:raise ValueError('72independent registered families required')
    a.output.mkdir(parents=True,exist_ok=False);tasks=[]
    for row in selected:
        source=row['source'];identity=row['id'];key=identity.split(':')[1];image=a.acquired/key/'preview.png'
        if digest(image)!=row['preview.png_sha256']:raise ValueError('source preview changed')
        width,height=row['header']['size']
        task=dict(id=identity,role=source['role'],family_id=source['family_id'],cell=source['cell'],
            width=width,height=height,purpose='editorial',model='gpt-6.1-terra',ratios=RATIOS,
            subject_intent='Choose the primary photographic person or coherent subgroup and preserve the event/story; complete exclusion of secondary people is allowed. Do not force every person into a crop.',
            image=dict(path=str(image),sha256=digest(image)),training_eligible=False,face_safety_certified=False,
            source_information=source['source_information'])
        path=a.output/(key+'.json');path.write_text(json.dumps(task,indent=2)+'\n');tasks.append(str(path))
    (a.output/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    protocol=dict(sources=72,roles=dict(Counter(r['source']['role'] for r in selected)),families=72,
        cells=cells,fit_per_cell=4,dev_per_cell=2,source_roles_changed=False,test_pixels_opened=False,
        selector='sha256(subject-choice-v4-real|source-id) within frozen role and count/size cell',
        acquisition_packet_sha256=digest(a.acquired/'SHA256SUMS.json'),review_mask_sha256=digest(a.review/'source-mask.json'),
        request_count=72,decisions_per_request=18,labels_admitted=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(protocol))

if __name__=='__main__':main()

