#!/usr/bin/env python3
"""Source-only measured face/pose regions with unmatched-person upper fallback."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import write
from refinement_inputs import local_heads
from person_risk_policy import person_count
from roi_geometry import person_views
from subject_origin_projection import padded_heads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','objects','faces','pose','original','previous','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    fixed=[(a.cache,'3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4'),(a.objects/'manifest.json','eb83a50e5c6760727b495d4d2e28b1f9d8c5fa23b1f844b3fc65c9beec0d1fa7'),(a.objects/'full/tf-efficientdet_lite2/results.json','e5cf955399185392c7413aa8620ddb5113aa2f65a1bd6a31f75a3f40a6bbd3a9'),(a.previous/'SHA256SUMS.json','6c8dada6dc31c97770933c1dd50a00ea73586e64baae665adb3dce18bef9b165')]
    if any(sha(path)!=digest for path,digest in fixed):raise ValueError('registered source/object/previous packets differ')
    verify_member(a.previous,'geometry/sources.json')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);old=json.loads((a.previous/'geometry/sources.json').read_text())['old']
    heads,head_ids=local_heads({'new':{'data':c},'old':{'data':{'sources':old}}},a.faces,a.pose,a.original)
    verify_member(a.pose,'sources.json');posed={s['id']:s for s in json.loads((a.pose/'sources.json').read_text())}
    keys=['id','role','family_id','native_size','working_size','rgb_sha256','png_sha256']
    if any(any(s[k]!=posed[s['id']][k] for k in keys) for s in c['sources']):raise ValueError('pose source identities differ')
    manifest=json.loads((a.objects/'manifest.json').read_text());images={r['id']:r for r in manifest['images']};detected={r['id']:r['result']['detections'] for r in json.loads((a.objects/'full/tf-efficientdet_lite2/results.json').read_text())['rows']}
    if manifest['confidence_threshold']!=.3 or manifest['max_results']!=50:raise ValueError('frozen person settings differ')
    variants={'face-body':{},'face-pose-body':{}};details=[]
    for source in c['sources']:
        sid=source['id'];w,h=source['native_size'];count=person_count(source,images[sid],detected[sid]);bodies=[]
        for detection in detected[sid]:
            if any(v['name']=='person' and v['score']>=.3 for v in detection['categories']):bodies.append(dict(person_views(detection['xywh'],source['working_size'],source['native_size'])))
        if len(bodies)!=count:raise ValueError('person/source mismatch')
        for policy,family in [('face-body','faces'),('face-pose-body','faces-pose')]:
            measured=heads['new'][sid][family];protected=padded_heads(w,h,measured,25);fallback=[]
            for body in bodies:
                b=body['person']
                if not any(b[0]*2<=head[0]+head[2]<=b[2]*2 and b[1]*2<=head[1]+head[3]<=b[3]*2 for head in measured):fallback.append(body['upper'])
            boxes=[list(b) for b in sorted(set(map(tuple,[*protected,*fallback])))];variants[policy][sid]=boxes
            details.append(dict(id=sid,role=source['role'],policy=policy,measured_heads=len(measured),person_detections=count,fallbacks=len(fallback),protected_regions=len(boxes)))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'regions.json',dict(schema='source-head-body-fallback-v1',sources=c['sources'],variants=variants,inputs={**{str(path):digest for path,digest in fixed},**head_ids},api_calls=0,new_inference=False,test_inference=False))
    write(a.output/'summary.json',dict(sources=72,policies=list(variants),details=details,regions_sha256=sha(a.output/'regions.json')))
    print(json.dumps(dict(sources=72,regions_sha256=sha(a.output/'regions.json'),fallbacks={name:sum(r['fallbacks'] for r in details if r['policy']==name) for name in variants})))


if __name__=='__main__':main()
