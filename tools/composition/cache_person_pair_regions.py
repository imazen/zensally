#!/usr/bin/env python3
"""Cached automatic full-person context descriptors; no chosen-subject inputs."""
import argparse
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from person_risk_policy import person_count
from roi_geometry import person_views
from distill_attention_linear import write
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from region_context import descriptor,FEATURES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);manifest=a.objects/'manifest.json';results=a.objects/'full/tf-efficientdet_lite2/results.json';expected=[(a.cache,'3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4'),(manifest,'eb83a50e5c6760727b495d4d2e28b1f9d8c5fa23b1f844b3fc65c9beec0d1fa7'),(results,'e5cf955399185392c7413aa8620ddb5113aa2f65a1bd6a31f75a3f40a6bbd3a9')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('pinned person/source inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);m=json.loads(manifest.read_text());images={r['id']:r for r in m['images']};objects={r['id']:r['result']['detections'] for r in json.loads(results.read_text())['rows']}
    if m['confidence_threshold']!=.3 or m['max_results']!=50:raise ValueError('registered detector settings differ')
    values={name:[] for name in ['person-all','person-largest']};records=[];positive=0
    for s in c['sources']:
        sid=s['id'];count=person_count(s,images[sid],objects[sid]);boxes=[];nw,nh=s['native_size']
        for detection in objects[sid]:
            if not any(v['name']=='person' and v['score']>=.3 for v in detection['categories']):continue
            name,b=person_views(detection['xywh'],s['working_size'],s['native_size'])[0]
            if name!='person':raise ValueError('owned full-person projection contract differs')
            boxes.append([b[0]/nw,b[1]/nh,b[2]/nw,b[3]/nh])
        if len(boxes)!=count or s['role'] not in ['fit','dev']:raise ValueError('source/count/role differs')
        positive+=bool(boxes);ordered=sorted(set(map(tuple,boxes)),key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),*b));largest=[list(ordered[0])] if ordered else []
        for name,b in [('person-all',boxes),('person-largest',largest)]:
            vector,detail=descriptor(b);values[name].append(vector);records.append(dict(id=sid,role=s['role'],family_id=s['family_id'],variant=name,boxes=b,**detail))
    if positive!=70:raise ValueError('frozen person-positive source count differs')
    a.output.mkdir(parents=True,exist_ok=False);torch.save(dict(schema='local-person-region-descriptors-v1',sources=c['sources'],variants={k:torch.tensor(v,dtype=torch.float32) for k,v in values.items()},features=FEATURES),a.output/'features.pt');write(a.output/'regions.json',records);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},features=FEATURES,teacher_annotation_inputs=False,selected_subject_inputs=False,crop_window_inputs=False,api_calls=0,test_inference=False,new_detector_inference=False,model_promoted=False));summary=dict(sources=72,person_positive=positive,features=FEATURES,features_sha256=sha(a.output/'features.pt'),omitted_slots={name:sum(r['omitted_slots'] for r in records if r['variant']==name) for name in values});write(a.output/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
