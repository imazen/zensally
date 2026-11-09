#!/usr/bin/env python3
"""Audit raw and rasterized subject-choice proposals without admitting labels."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'composition'))
from subject_policy import face_relation


def pixels(box,width,height,*,outward=False):
    values=[box[k]*scale/100 for k,scale in zip(['left','top','right','bottom'],[width,height,width,height])]
    if outward:values=[math.floor(values[0]),math.floor(values[1]),math.ceil(values[2]),math.ceil(values[3])]
    return values


def project(box,width,height,ratio,mode):
    rw,rh=map(int,ratio.split(':'));maximum=min(width//rw,height//rh)
    if maximum<1:raise ValueError('source cannot represent requested integer aspect')
    raw=pixels(box,width,height)
    scale=maximum if mode=='minimal' else max(1,min(maximum,round(min((raw[2]-raw[0])/rw,(raw[3]-raw[1])/rh))))
    w,h=scale*rw,scale*rh
    x=max(0,min(width-w,round((raw[0]+raw[2]-w)/2)))
    y=max(0,min(height-h,round((raw[1]+raw[3]-h)/2)))
    return [x,y,x+w,y+h]


def audit(box,rois,selected,excluded,width,height,*,outward=True):
    faces=[r for r in rois if r['kind']=='face_head']
    relations={r['id']:face_relation(box,pixels(r['box'],width,height,outward=outward)) for r in faces}
    missing=[sid for sid in selected if any(r['id']==sid and r['kind'] in ['person','animal'] for r in rois)
        and not any(r['owner_subject_id']==sid for r in faces)]
    selected_bad=[r['id'] for r in faces if r['owner_subject_id'] in selected and relations[r['id']]!='contained']
    excluded_bad=[r['id'] for r in faces if r['owner_subject_id'] in excluded and relations[r['id']]!='excluded']
    cut=[rid for rid,state in relations.items() if state=='cut']
    return dict(face_relations=relations,cut_faces=cut,selected_faces_not_contained=selected_bad,
        excluded_faces_not_excluded=excluded_bad,selected_people_without_face_roi=missing,
        geometry_compatible=not(cut or selected_bad or excluded_bad or missing),
        complete_face_coverage_reviewed=False,training_eligible=False)


def inspect(zones,width,height):
    rows=[]
    for request in zones['crops']:
        for index,option in enumerate(request['options'] if request['action']!='contain' else []):
            raw=pixels(option['box'],width,height)
            projected=project(option['box'],width,height,request['ratio'],request['mode'])
            args=(zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],width,height)
            rows.append(dict(ratio=request['ratio'],mode=request['mode'],option=index,action=request['action'],
                raw_xyxy=raw,projected_xyxy=projected,edge_adjustment_pixels=[a-b for a,b in zip(projected,raw)],
                raw=audit(raw,*args,outward=False),projected=audit(projected,*args),teacher_face_check=option['face_policy_check']))
    return dict(rows=rows,options=len(rows),raw_geometry_failures=sum(not r['raw']['geometry_compatible'] for r in rows),
        projected_geometry_failures=sum(not r['projected']['geometry_compatible'] for r in rows),
        training_eligible=False,face_safety_certified=False,
        note='Compatibility is only relative to supplied unreviewed face boxes; exact-aspect projection is audited separately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('zones',type=Path);p.add_argument('width',type=int);p.add_argument('height',type=int);p.add_argument('output',type=Path);a=p.parse_args()
    result=inspect(json.loads(a.zones.read_text()),a.width,a.height)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
