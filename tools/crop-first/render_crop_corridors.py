#!/usr/bin/env python3
"""Render endpoint/quartile controls without admitting an unreviewed interval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from crop_corridors import samples,member
from rendered_pixels import render
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'composition'))
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','case_id','result','native','resizer','encoder','output']:p.add_argument(key,type=Path if key!='case_id' else str)
    p.add_argument('--audit-uncertain',action='store_true');a=p.parse_args()
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned executable differs')
    plan=json.loads((a.root/'plan.json').read_text());case=next(r for r in plan['cases'] if r['case_id']==a.case_id);reply=json.loads(a.result.read_text())
    if not reply['annotation_valid']:raise ValueError('structurally invalid source ranges cannot be rendered by this caller')
    zones=json.loads((Path(reply['entry'])/'zones.json').read_text())
    if zones['source_id']!=case['id']:raise ValueError('source ranges differ')
    source=next(r for r in json.loads((a.native/'manifest.json').read_text())['images'] if r['source_id']==case['id'])
    raw=Path(source['rgb']).read_bytes()
    if sha(Path(source['rgb']))!=source['rgb_sha256']:raise ValueError('owned RGB changed')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'tasks').mkdir();groups=[];tasks=[]
    for index,pair in enumerate(zones['projected_ranges']):
        if pair.get('pair_protocol',1)==2 and not pair['outer_is_maximum_area']:continue
        if not pair['ready_for_rendered_review'] and not(a.audit_uncertain and pair['geometry']['geometrically_admissible'] and pair['teacher_claims_excellent'] and pair['inner']!=pair['outer']):continue
        name=a.case_id+'-pair'+str(index);directory=a.output/name;directory.mkdir();candidates=[]
        boxes=samples(pair['inner'],pair['outer'],pair['ratio'])
        for j,box in enumerate([*boxes,None]):
            if box is not None and not member(box,pair['inner'],pair['outer'],pair['ratio']):raise ValueError('sample outside exact crop interval')
            origin=['inner','quarter25','midpoint','quarter75','outer','contain'][j]
            image=render(raw,source['width'],source['height'],source['width']*3,[case['width'],case['height']],box,pair['ratio'],directory/origin,a.resizer,a.encoder)
            candidates.append(dict(image,origin=origin,native_box=box))
        group=dict(case_id=name,source_id=case['id'],role=case['role'],family_id=case['family_id'],ratio=pair['ratio'],pair=pair,candidates=candidates,quality_admitted=False,training_eligible=False,face_coverage_certified=False)
        groups.append(group)
        for presentation,model in [('terra-first','gpt-6.1-terra'),('sol-review','gpt-6.1-sol'),('sol-repeat','gpt-6.1-sol')]:
            order=sorted(candidates,key=lambda c:hashlib.sha256((name+'|'+presentation+'|'+c['origin']).encode()).hexdigest())
            shown=[dict(c,id=letter) for letter,c in zip('ABCDEF',order)]
            task=dict(id=name+'-'+presentation,case_id=name,presentation_id=presentation,model=model,role=case['role'],family_id=case['family_id'],
                width=case['width'],height=case['height'],ratio=pair['ratio'],mode='maximal',image=case['image'],candidates=shown,training_eligible=False)
            path=a.output/'tasks'/(task['id']+'.json');path.write_text(json.dumps(task,indent=2)+'\n');tasks.append(str(path))
    (a.output/'plan.json').write_text(json.dumps(dict(groups=groups,task_paths=tasks,proposer_result_sha256=sha(a.result),audit_uncertain=a.audit_uncertain,training_eligible=False),indent=2)+'\n')
    print(json.dumps(dict(source=case['id'],groups=len(groups),requests=len(tasks),range_proposals=len(zones['projected_ranges']),source_inventory=zones['face_inventory'])))


if __name__=='__main__':main()
