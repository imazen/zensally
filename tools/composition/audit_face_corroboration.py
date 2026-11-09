#!/usr/bin/env python3
"""Measure cached face-family/person support without discarding uncertain faces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from artifact_identity import sha
from subject_tiled_faces import iou


def family(model):
    if model.startswith('native-yunet'):return 'yunet'
    if model=='native-ultraface':return 'ultraface'
    if model in ['float16-blaze_face_short_range','float16-blaze_face_full_range_sparse','float16-face_landmarker']:return 'mediapipe-face-stack'
    raise ValueError('unregistered face family: '+model)


def inside(center, box):
    return box[0]<=center[0]<=box[2] and box[1]<=center[1]<=box[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['faces','face_manifest','direct','original','miap_objects','unsplash_objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.direct)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472' or sha(a.faces/'SHA256SUMS.json')!='c7872a992e607cb062089fdd678a0629a16d9f86841c831a4020cd9b8a579b29' or sha(a.original/'SHA256SUMS.json')!='0c1909193ec1b219adb22a005cfbd1e32d09929e6890ef0d6e8820c28b8f54e5':raise ValueError('frozen proposal parents differ')
    direct=torch.load(a.direct,map_location='cpu',weights_only=False)
    nominated={s['id']:[list(r[k]/100 for k in ['left','top','right','bottom']) for r in direct['head_rois'][i]] for i,s in enumerate(direct['sources']) if s['annotation_valid']}
    real={s['id']:s for s in json.loads(a.face_manifest.read_text())['images']}
    if set(real)!={s['id'] for s in direct['sources']}:raise ValueError('direct/face source set differs')
    original={s['source_id']:s for s in json.loads((a.original/'manifest.json').read_text())['images']}
    reviewed={r['id'] for r in json.loads((a.original/'witnesses/visual-review.json').read_text())['cases']}
    if len(reviewed)!=3:raise ValueError('fixed three reviewed false-positive sources required')
    panels={'real72':(real,a.faces/'analysis/boxes.json',a.miap_objects),
            'original25':(original,a.original/'regions/boxes.json',a.unsplash_objects)}
    a.output.mkdir(parents=True,exist_ok=False);proposals=[];head_rows=[];protocol={};challenge=[]
    for panel,(sources,face_path,object_root) in panels.items():
        object_manifest=object_root/'manifest.json';object_path=object_root/'full/tf-efficientdet_lite2/results.json'
        objdoc=json.loads(object_manifest.read_text());images={s['id']:s for s in objdoc['images']};objects={r['id']:r['result']['detections'] for r in json.loads(object_path.read_text())['rows']}
        if objdoc['confidence_threshold']!=.3 or objdoc['max_results']!=50:raise ValueError('existing person threshold differs')
        local=defaultdict(list)
        for row in json.loads(face_path.read_text()):
            if row['model']=='float16-pose_landmarker_lite':continue
            for box in row['boxes']:local[row['id']].append(dict(model=row['model'],family=family(row['model']),box=box))
        protocol[panel]=dict(face_boxes_sha256=sha(face_path),object_manifest_sha256=sha(object_manifest),object_results_sha256=sha(object_path))
        for sid,source in sources.items():
            if source['rgb_sha256']!=images[sid]['rgb_sha256'] or [source['width'],source['height']]!=[images[sid]['width'],images[sid]['height']]:raise ValueError('person and face pixels differ')
            if panel=='real72' and source['rgb_sha256']!=next(s['rgb_sha256'] for s in direct['sources'] if s['id']==sid):raise ValueError('weak head source pixels differ')
            w,h=source['width'],source['height'];people=[]
            for d in objects[sid]:
                if any(c['name']=='person' and c['score']>=.3 for c in d['categories']):
                    x,y,cw,ch=d['xywh'];people.append([x/w,y/h,(x+cw)/w,(y+ch)/h])
            values=local[sid]
            for index,value in enumerate(values):
                b=value['box'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
                value.update(person_supported=any(inside(center,p) for p in people),other_family_max_iou=max((iou(b,r['box']) for r in values if r['family']!=value['family']),default=0.))
                proposals.append(dict(panel=panel,id=sid,role=source['role'],index=index,**value))
            for threshold in [.1,.3,.5]:
                groups={'all':values,'other-family':[v for v in values if v['other_family_max_iou']>=threshold],
                    'person':[v for v in values if v['person_supported']],
                    'both':[v for v in values if v['person_supported'] and v['other_family_max_iou']>=threshold],
                    'either':[v for v in values if v['person_supported'] or v['other_family_max_iou']>=threshold]}
                for rule,kept in groups.items():
                    if panel=='real72' and sid in nominated:
                        heads=nominated[sid];centers=[[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in heads]
                        head_rows.append(dict(id=sid,role=source['role'],threshold=threshold,rule=rule,heads=len(heads),
                            covered_centers=sum(any(inside(p,v['box']) for v in kept) for p in centers),
                            contained_heads=sum(any(v['box'][0]<=b[0] and v['box'][1]<=b[1] and v['box'][2]>=b[2] and v['box'][3]>=b[3] for v in kept) for b in heads),
                            proposals=len(kept),person_proposals=len(people)))
                    if panel=='original25' and sid in reviewed:challenge.append(dict(id=sid,threshold=threshold,rule=rule,proposals=len(kept),person_proposals=len(people),source=source['source']))
    summary=[]
    for role in ['fit','dev']:
        for threshold in [.1,.3,.5]:
            for rule in ['all','other-family','person','both','either']:
                selected=[r for r in head_rows if r['role']==role and r['threshold']==threshold and r['rule']==rule]
                summary.append(dict(role=role,threshold=threshold,rule=rule,sources=len(selected),**{k:sum(r[k] for r in selected) for k in ['heads','covered_centers','contained_heads','proposals']}))
    protocol.update(primary_iou=.1,sensitivity_iou=[.3,.5],unsupported='uncertain, never a non-face label',api_calls=0,test_inference=False,model_promoted=False,face_coverage_certified=False)
    for name,value in [('protocol',protocol),('summary',dict(summary=summary,reviewed_negative_challenge=challenge,api_calls=0,test_inference=False,model_promoted=False)),('proposal-signals',proposals),('head-coverage',head_rows)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(primary=[r for r in summary if r['threshold']==.1],challenge=[r for r in challenge if r['threshold']==.1])))


if __name__=='__main__':main()
