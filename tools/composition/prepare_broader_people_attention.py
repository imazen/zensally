#!/usr/bin/env python3
"""Reuse the exact owned gaze/scan preprocessing and226 pinned local assets."""
import argparse
import json
from pathlib import Path
import subprocess
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','regions','template','resizer','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    fixed=[(a.plan,'1b2da79fe0a04d6a73121529d9e2c8940052cfb1812ccadcfc0fd70f01814e4a'),(a.regions,'72346b4a8ec6875eb047aa393b177c6aeb3c665b2eb49a0678e6310c1df4523a'),(a.template,'1b4a0e04f7011c6d82a9530d838d85b76abf2dae5e4bd657434442590ea33673'),(a.resizer,'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041')]
    if any(sha(path)!=digest for path,digest in fixed):raise ValueError('fixed broader teacher inputs differ')
    template=json.loads(a.template.read_text());members=template['teacher_members']
    if len(members)!=226 or any(sha(Path(path))!=digest for path,digest in members.items()):raise ValueError('complete pinned local teacher subtree differs')
    plan=json.loads(a.plan.read_text());regions=json.loads(a.regions.read_text())
    if plan['sources']!=regions['sources']:raise ValueError('broader source identities differ')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'inputs').mkdir();images=[];heads={};omissions=[]
    for source in plan['sources']:
        row=dict(source);w,h=source['working_size'];row.update(width=w,height=h);raw=Path(source['rgb']).read_bytes()
        if sha(Path(source['rgb']))!=source['rgb_sha256'] or len(raw)!=w*h*3:raise ValueError('owned source pixels differ')
        scale=max(128/min(w,h),512/max(w,h));dw,dh=round(w*scale),round(h*scale)
        if dw*dh>1024**2:raise ValueError('registered one-megapixel scan bound exceeded')
        output=subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),f'0,0,{w},{h},448,448;0,0,{w},{h},{dw},{dh}'],input=raw);offset=448*448*3
        if len(output)!=offset+dw*dh*3:raise ValueError('resized input lengths differ')
        for name,value,rw,rh in [('gaze',output[:offset],448,448),('scan',output[offset:],dw,dh)]:
            path=a.output/'inputs'/(source['id']+'-'+name+'.rgb8')
            with path.open('xb') as f:f.write(value)
            row[name]=dict(path=str(path),width=rw,height=rh,sha256=sha(path))
        proposals=[]
        for x0,y0,x1,y1 in regions['face_models'][source['id']]['native-yunet-tiled-interior']:
            bw,bh=x1-x0,y1-y0;proposals.append([max(0.,x0-.15*bw),max(0.,y0-.25*bh),min(1.,x1+.15*bw),min(1.,y1+.25*bh)])
        proposals.sort(key=lambda b:-(b[2]-b[0])*(b[3]-b[1]));heads[source['id']]={k:source[k] for k in ['role','family_id','rgb_sha256']};heads[source['id']]['people']=[dict(head=b,proposal_confidence_available=False) for b in proposals[:16]];omissions.append(dict(id=source['id'],input_proposals=len(proposals),retained=min(16,len(proposals)),omitted=max(0,len(proposals)-16)));images.append(row)
    for name,value in [('heads.json',heads),('inputs.json',dict(images=images,teacher_members=members,inputs={str(path):digest for path,digest in fixed},omissions=omissions,preprocessing='same owned448-square gaze; aspect-preserving scan min128/max>=512, four six-fixation paths',api_calls=0,test_inference=False))]:
        with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(sources=16,head_sources=sum(bool(v['people']) for v in heads.values()),omitted_heads=sum(r['omitted'] for r in omissions),input_bytes=sum(p.stat().st_size for p in (a.output/'inputs').iterdir()),verified_teacher_members=226)))


if __name__=='__main__':main()
