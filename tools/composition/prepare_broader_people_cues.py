#!/usr/bin/env python3
"""Source-bound local face/pose inputs for the frozen broader DEV panel."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha
from prepare_subject_face_cues import MODELS
from subject_native_faces import BINARY_SHA
from subject_tiled_faces import windows,extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','template','binary','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.plan)!='1b2da79fe0a04d6a73121529d9e2c8940052cfb1812ccadcfc0fd70f01814e4a' or sha(a.binary)!=BINARY_SHA:raise ValueError('frozen plan/native binary differs')
    plan=json.loads(a.plan.read_text());template=json.loads(a.template.read_text());models=template['models']
    if {m['id'] for m in models}!=set(MODELS) or any(m['sha256']!=MODELS[m['id']] or sha(Path(m['path']))!=m['sha256'] for m in models):raise ValueError('same four pinned local teachers required')
    a.output.mkdir(parents=True,exist_ok=False);images=[];native=[];tiles=[];(a.output/'tiles/rgb').mkdir(parents=True)
    for source in plan['sources']:
        if source['role']!='dev':raise ValueError('DEV only; TEST closed')
        rgb=Path(source['rgb']);w,h=source['working_size']
        if sha(rgb)!=source['rgb_sha256'] or rgb.stat().st_size!=w*h*3 or sha(Path(source['png']))!=source['png_sha256']:raise ValueError('owned source pixels differ')
        row=dict(id=source['id'],source_id=source['id'],family_id=source['family_id'],role='dev',width=w,height=h,native_size=source['native_size'],rgb=str(rgb),rgb_sha256=source['rgb_sha256'],display=source['png'],benchmark=False);images.append(row);native.append(dict(row,id=row['id'].replace(':','-')));raw=rgb.read_bytes()
        for index,box in enumerate(windows(w,h)):
            identifier=source['id'].replace(':','-')+'-tile'+str(index);path=a.output/'tiles/rgb'/(identifier+'.rgb')
            with path.open('xb') as f:f.write(extract(raw,w,h,box))
            tiles.append(dict(id=identifier,source_id=source['id'],role='dev',rgb=str(path),rgb_sha256=sha(path),width=box[2]-box[0],height=box[3]-box[1],source_box=box,source_size=[w,h],benchmark=False))
    for name,rows,assets in [('mediapipe',images,models),('native',native,[]),('tiles',tiles,[])]:
        root=a.output/name;root.mkdir(exist_ok=name=='tiles')
        with (root/'manifest.json').open('x') as f:json.dump(dict(images=rows,models=assets,source_plan_sha256=sha(a.plan),native_binary_sha256=BINARY_SHA,face_confidence=.5,pose_count=8,face_landmark_count=16,test_inference=False,api_calls=0),f,indent=2);f.write('\n')
        with (root/'inputs.tsv').open('x') as f:f.write(''.join(f"{r['id']}\t{r['width']}\t{r['height']}\t{r['rgb']}\t0\n" for r in rows))
    print(json.dumps(dict(sources=len(images),mediapipe_calls=len(images)*4,native_full_calls=len(native)*2,tile_calls=len(tiles),tile_bytes=sum(p.stat().st_size for p in (a.output/'tiles/rgb').iterdir()))))


if __name__=='__main__':main()
