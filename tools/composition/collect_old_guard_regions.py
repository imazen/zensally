#!/usr/bin/env python3
"""Collect fixed local face/pose geometry on the original25 DEV sources."""
import argparse
import json
import math
from pathlib import Path
from artifact_identity import sha
from subject_tiled_faces import map_tile_faces, merge
from pose_head_bounds import head_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    manifest=json.loads((root/'manifest.json').read_text());sources={s['id']:s for s in manifest['images']}
    boxes=[];poses=[];identities={}
    def rows(path, wrapped=False):
        identities[str(path.relative_to(root))]=sha(path)
        value=json.loads(path.read_text());value=value['rows'] if wrapped else value
        result={r['id']:r['result'] for r in value}
        if len(value)!=25 or set(result)!=set(sources):raise ValueError('original25 source output inventory differs')
        return result
    def clipped(regions):
        output=[]
        for b in regions:
            if not all(math.isfinite(v) for v in b):raise ValueError('nonfinite local box')
            b=[max(0.,min(1.,float(v))) for v in b]
            if not(b[0]<b[2] and b[1]<b[3]):raise ValueError('empty local box')
            output.append(b)
        return output
    native={model:rows(root/'native'/model/'results.json') for model in ['ultraface','yunet']}
    proposals={sid:[dict(box=[max(0.,min(1.,v/100)) for v in f['box_percent']],confidence=f['confidence'],origin='full',internal_edge=False)
                    for f in value['faces']] for sid,value in native['yunet'].items()}
    for model,values in native.items():
        for sid,value in values.items():boxes.append(dict(id=sources[sid]['source_id'],role='dev',model='native-'+model,boxes=clipped([[v/100 for v in f['box_percent']] for f in value['faces']])))
    tiles={r['id']:r for r in json.loads((root/'tiles/manifest.json').read_text())['images']}
    path=root/'tiles/native/yunet/results.json';raw=json.loads(path.read_text());identities[str(path.relative_to(root))]=sha(path)
    if len(raw)!=100 or {r['id'] for r in raw}!=set(tiles):raise ValueError('all100 tile evaluations required')
    safe_ids={s['source_id']:sid for sid,s in sources.items()}
    for r in raw:
        tile=tiles[r['id']]
        proposals[safe_ids[tile['source_id']]].extend(map_tile_faces(tile,r['result']['faces']))
    for model in ['native-yunet-tiled','native-yunet-tiled-interior']:
        for sid,values in proposals.items():
            chosen=merge([v for v in values if model=='native-yunet-tiled' or not v['internal_edge']])
            boxes.append(dict(id=sources[sid]['source_id'],role='dev',model=model,boxes=clipped([v['box'] for v in chosen])))
    for asset in manifest['models']:
        model=asset['id'];values=rows(root/'mediapipe'/model/'results.json',True)
        for sid,value in values.items():
            source=sources[sid];w,h=source['width'],source['height'];regions=[]
            if 'blaze_face' in model:
                for d in value['detections']:
                    x,y,cw,ch=d['xywh'];regions.append([x/w,y/h,(x+cw)/w,(y+ch)/h])
            elif 'face_landmarker' in model:
                for points in value['landmarks']:
                    regions.append([min(p['x'] for p in points),min(p['y'] for p in points),max(p['x'] for p in points),max(p['y'] for p in points)])
            elif 'pose_landmarker' in model:
                nw,nh=source['native_size']
                for index,points in enumerate(value['landmarks']):
                    projected=[dict(p,x=p['x']*nw,y=p['y']*nh) for p in points]
                    box,reason=head_bounds(projected,nw,nh)
                    poses.append(dict(id=source['source_id'],index=index,native_box=box,reason=reason))
                continue
            else:raise ValueError('unregistered model')
            boxes.append(dict(id=source['source_id'],role='dev',model=model,boxes=clipped(regions)))
    output=root/'regions';output.mkdir(exist_ok=False)
    (output/'boxes.json').write_text(json.dumps(boxes,indent=2)+'\n')
    (output/'pose-heads.json').write_text(json.dumps(poses,indent=2)+'\n')
    (output/'identity.json').write_text(json.dumps(dict(inputs_sha256=identities,manifest_sha256=sha(root/'manifest.json'),api_calls=0,test_inference=False),indent=2)+'\n')
    print(json.dumps(dict(face_model_rows=len(boxes),pose_estimates=len(poses),accepted_pose_estimates=sum(r['native_box'] is not None for r in poses))))


if __name__=='__main__':main()
