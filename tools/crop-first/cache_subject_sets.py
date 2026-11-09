#!/usr/bin/env python3
"""Audit fixed-bank coverage and cache weak acceptable windows; never admit faces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from subject_geometry import audit,pixels,project
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'composition'))
from artifact_identity import sha
from subject_policy import face_relation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['tasks','responses','rank_cache','output']:p.add_argument(key,type=Path)
    a=p.parse_args();status=json.loads((a.responses/'PROGRESS.json').read_text())
    if status.get('stage')!='complete' or status['completed']!=72 or status.get('errors'):raise ValueError('all72 registered reference attempts must finish without orchestration errors')
    if sha(a.rank_cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d':raise ValueError('fixed broader rank cache differs')
    tasks=[json.loads(Path(path).read_text()) for path in json.loads(a.tasks.read_text())]
    base=torch.load(a.rank_cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(base['sources'])}
    a.output.mkdir(parents=True,exist_ok=False);indices=[];sources=[];positive=[];negative=[];references=[];reference_valid=[];records=[];counts=Counter()
    for task in tasks:
        i=lookup[task['id']];source=base['sources'][i]
        if source['role']!=task['role'] or source['family_id']!=task['family_id'] or source['native_size']!=[task['width'],task['height']]:raise ValueError('reference/cache source join differs')
        result=json.loads((a.responses/(task['id'].replace(':','-')+'.json')).read_text());entry=Path(result['entry'])
        for name,digest in json.loads((entry/'hashes.json').read_text()).items():
            if sha(entry/name)!=digest:raise ValueError('cached reference bytes differ')
        indices.append(i);sources.append(dict(source,ratios=base['ratios'],annotation_valid=result['annotation_valid'],reference_cache_key=result['cache_key']))
        pos=np.zeros((22,128),bool);neg=np.zeros_like(pos);refs=np.zeros((22,2,4),np.float32);known=np.zeros((22,2),bool)
        counts[task['role']+'_sources']+=1
        if result['annotation_valid']:
            zones=json.loads((entry/'zones.json').read_text());w,h=task['width'],task['height'];scale=np.array([w,h,w,h])
            faces=[pixels(r['box'],w,h,outward=True) for r in zones['rois'] if r['kind']=='face_head']
            for request in zones['crops']:
                ri=base['ratios'].index(request['ratio']);q=ri*2+(request['mode']=='maximal')
                native=base['native_boxes'][i][ri];boxes=np.asarray(native,float)
                eligible=base['eligible'][i,q,:len(native)].numpy()
                cut=np.array([any(face_relation(box.tolist(),face)=='cut' for face in faces) for box in boxes])
                neg[q,:len(native)]=cut&eligible
                counts['decisions']+=1;counts['action_'+request['action']]+=1
                good=[]
                for option in request['options'] if request['action']!='contain' else []:
                    raw=pixels(option['box'],w,h);exact=project(option['box'],w,h,request['ratio'],request['mode'])
                    args=(zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)
                    reason=None
                    if option['face_policy_check']!='whole_faces_or_excluded':reason='teacher_uncertain'
                    elif not audit(raw,*args,outward=False)['geometry_compatible']:reason='raw_face_geometry'
                    elif not audit(exact,*args)['geometry_compatible']:reason='projected_face_geometry'
                    elif max(np.abs(np.asarray(exact)-raw)/scale*128)>3:reason='projection_moves_more_than3grid'
                    if reason:counts['rejected_'+reason]+=1;continue
                    good.append((np.asarray(exact)/scale,option))
                for slot,(target,option) in enumerate(good[:2]):
                    refs[q,slot]=target;known[q,slot]=True
                    close=(np.abs(boxes/scale-target)*128).max(-1)<=3
                    safe=np.array([audit(box.tolist(),zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)['geometry_compatible'] for box in boxes])
                    pos[q,:len(native)]|=close&safe&eligible
                if np.any(pos[q]&neg[q]):raise ValueError('contradictory positive and explicit face-cut labels')
                has_reference=bool(known[q].any());covered=bool(pos[q].any())
                counts['accepted_reference_queries']+=has_reference;counts['covered_reference_queries']+=covered
                counts['contrastive_queries']+=covered and bool(neg[q].any())
                records.append(dict(id=task['id'],role=task['role'],family_id=task['family_id'],ratio=request['ratio'],mode=request['mode'],
                    action=request['action'],accepted_options=int(known[q].sum()),bank_covered=covered,
                    positive_candidates=int(pos[q].sum()),explicit_face_cut_candidates=int(neg[q].sum()),eligible_candidates=int(eligible.sum())))
        else:counts['invalid_responses']+=1
        positive.append(pos);negative.append(neg);references.append(refs);reference_valid.append(known)
    output=dict(schema='subject-choice-fixed-bank-v1',sources=sources,ratios=base['ratios'],
        image=base['image'][indices],content=base['content'][indices],boxes=base['boxes'][indices],eligible=base['eligible'][indices],
        positive=torch.from_numpy(np.array(positive)),negative=torch.from_numpy(np.array(negative)),
        references=torch.from_numpy(np.array(references)),reference_valid=torch.from_numpy(np.array(reference_valid)),
        protocol=dict(source_tasks_sha256=sha(a.tasks),parent_rank_cache_sha256=sha(a.rank_cache),reference_tolerance_grid=3,
            input_size=256,source_roles_unchanged=True,test_inference=False,labels='weak unreviewed teacher regions/windows',
            unknown_candidates_are_not_negatives=True,face_coverage_certified=False,shipping_guards_unchanged=True))
    torch.save(output,a.output/'cache.pt')
    (a.output/'queries.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(counts=counts,cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,
        coverage_note='Finite proposed alternatives, not exhaustive artistic acceptability; geometry is relative to supplied unreviewed ROIs.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()

