#!/usr/bin/env python3
"""Reproduce the seven-face union/pose/body proposal contract on broader DEV."""
import argparse
import json
import math
from pathlib import Path
from artifact_identity import sha
from subject_tiled_faces import map_tile_faces,merge
from pose_head_bounds import head_bounds
from roi_geometry import person_views
from person_risk_policy import person_count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','cues','objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.plan)!='1b2da79fe0a04d6a73121529d9e2c8940052cfb1812ccadcfc0fd70f01814e4a':raise ValueError('source plan changed')
    sources=json.loads(a.plan.read_text())['sources'];ids={s['id'] for s in sources};safe={s['id'].replace(':','-'):s['id'] for s in sources};identities={str(a.plan):sha(a.plan)}
    for relative,count in [('mediapipe/local/summary.json',4),('native/local/summary.json',2),('tiles/local/summary.json',1)]:
        path=a.cues/relative;values=json.loads(path.read_text())
        if len(values)!=count or any(r['returncode'] for r in values):raise ValueError('all local teachers must complete')
        identities[str(path)]=sha(path)
    def read(relative,wrapped=False,tiled=False):
        path=a.cues/relative;identities[str(path)]=sha(path);value=json.loads(path.read_text());rows=value['rows'] if wrapped else value
        if tiled:return rows
        rows={safe.get(r['id'],r['id']):r['result'] for r in rows}
        if set(rows)!=ids:raise ValueError('complete source outputs required')
        return rows
    by_source={s['id']:{} for s in sources}
    def normalized(regions):
        output=[]
        for region in regions:
            if not all(math.isfinite(v) for v in region):raise ValueError('nonfinite face proposal')
            box=[max(0.,min(1.,v)) for v in region]
            if not(box[0]<box[2] and box[1]<box[3]):raise ValueError('empty face proposal')
            output.append(box)
        return output
    native={model:read('native/local/'+model+'/results.json') for model in ['ultraface','yunet']};tiled={sid:[] for sid in ids}
    for model,rows in native.items():
        for sid,value in rows.items():
            boxes=normalized([[v/100 for v in f['box_percent']] for f in value['faces']]);by_source[sid]['native-'+model]=boxes
            if model=='yunet':tiled[sid]=[dict(box=b,confidence=f['confidence'],origin='full',internal_edge=False) for b,f in zip(boxes,value['faces'])]
    manifest=a.cues/'tiles/manifest.json';identities[str(manifest)]=sha(manifest);tiles={r['id']:r for r in json.loads(manifest.read_text())['images']};raw=read('tiles/local/yunet/results.json',tiled=True)
    if len(raw)!=64 or {r['id'] for r in raw}!=set(tiles):raise ValueError('all64 tiles required')
    for row in raw:
        tile=tiles[row['id']]
        if sha(Path(tile['rgb']))!=tile['rgb_sha256']:raise ValueError('tile pixels changed')
        tiled[tile['source_id']].extend(map_tile_faces(tile,row['result']['faces']))
    for sid,values in tiled.items():
        for name in ['native-yunet-tiled','native-yunet-tiled-interior']:by_source[sid][name]=[r['box'] for r in merge([r for r in values if name=='native-yunet-tiled' or not r['internal_edge']])]
    poses={sid:[] for sid in ids};reasons=[]
    for model in ['float16-blaze_face_short_range','float16-blaze_face_full_range_sparse','float16-face_landmarker','float16-pose_landmarker_lite']:
        rows=read('mediapipe/local/'+model+'/results.json',wrapped=True)
        for s in sources:
            sid=s['id'];w,h=s['working_size'];nw,nh=s['native_size'];value=rows[sid]
            if 'blaze_face' in model:regions=[[d['xywh'][0]/w,d['xywh'][1]/h,(d['xywh'][0]+d['xywh'][2])/w,(d['xywh'][1]+d['xywh'][3])/h] for d in value['detections']]
            elif model=='float16-face_landmarker':regions=[[min(p['x'] for p in points),min(p['y'] for p in points),max(p['x'] for p in points),max(p['y'] for p in points)] for points in value['landmarks']]
            else:
                for points in value['landmarks']:
                    box,reason=head_bounds([dict(p,x=p['x']*nw,y=p['y']*nh) for p in points],nw,nh);reasons.append(dict(id=sid,box=box,reason=reason))
                    if box is not None:poses[sid].append(box)
                continue
            by_source[sid][model]=normalized(regions)
    manifest=a.objects/'manifest.json';results=a.objects/'full/tf-efficientdet_lite2/results.json'
    if sha(manifest)!='eb83a50e5c6760727b495d4d2e28b1f9d8c5fa23b1f844b3fc65c9beec0d1fa7' or sha(results)!='e5cf955399185392c7413aa8620ddb5113aa2f65a1bd6a31f75a3f40a6bbd3a9':raise ValueError('cached person results differ')
    identities.update({str(manifest):sha(manifest),str(results):sha(results)});images={r['id']:r for r in json.loads(manifest.read_text())['images']};objects={r['id']:r['result']['detections'] for r in json.loads(results.read_text())['rows']};groups=[];counts=[]
    for s in sources:
        sid=s['id'];nw,nh=s['native_size'];group=dict(faces=[],pose=poses[sid],upper=[],person=[])
        if len(by_source[sid])!=7:raise ValueError('same seven face families required')
        for boxes in by_source[sid].values():group['faces'].extend([[max(0,math.floor(b[0]*nw)),max(0,math.floor(b[1]*nh)),min(nw,math.ceil(b[2]*nw)),min(nh,math.ceil(b[3]*nh))] for b in boxes])
        counts.append(person_count(s,images[sid],objects[sid]))
        for detection in objects[sid]:
            if any(v['name']=='person' and v['score']>=.3 for v in detection['categories']):
                for name,box in person_views(detection['xywh'],s['working_size'],s['native_size']):group[name].append(box)
        groups.append({k:[list(b) for b in sorted(set(map(tuple,v)))] for k,v in group.items()})
    order=sorted(range(16),key=lambda i:sources[i]['id']);donors=list(range(16))
    for k,i in enumerate(order):donors[i]=order[(k+1)%16]
    result=dict(schema='broader-source-regions-v1',sources=sources,groups=groups,donors=donors,person_counts=counts,face_models=by_source,pose_reasons=reasons,inputs=identities,api_calls=0,test_inference=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(sources=16,face_positive=sum(bool(g['faces']) for g in groups),pose_positive=sum(bool(g['pose']) for g in groups),person_positive=sum(v>0 for v in counts),regions_sha256=sha(a.output))))


if __name__=='__main__':main()
