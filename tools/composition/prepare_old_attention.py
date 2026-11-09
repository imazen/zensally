#!/usr/bin/env python3
"""Prepare owned-pixel attention inputs for original25 DEV, with pinned teachers."""
import argparse
import json
from pathlib import Path
import subprocess
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['guard','attention','resizer','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.guard/'SHA256SUMS.json')!='0c1909193ec1b219adb22a005cfbd1e32d09929e6890ef0d6e8820c28b8f54e5':raise ValueError('original guard packet differs')
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':raise ValueError('owned resizer differs')
    members=json.loads((a.guard/'SHA256SUMS.json').read_text())
    for name in ['manifest.json','regions/boxes.json','mediapipe/float16-pose_landmarker_lite/results.json']:
        if sha(a.guard/name)!=members[name]:raise ValueError('original input member differs')
    if sha(a.attention/'SHA256SUMS.json')!='86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3':raise ValueError('existing attention packet differs')
    frozen=json.loads((a.attention/'SHA256SUMS.json').read_text())
    if sha(a.attention/'inputs.json')!=frozen['inputs.json']:raise ValueError('canonical teacher input record differs')
    prior=json.loads((a.attention/'inputs.json').read_text());teachers=prior['teacher_members']
    if len(teachers)!=226 or any(sha(Path(path))!=digest for path,digest in teachers.items()):raise ValueError('complete pinned teacher subtree differs')
    rows=json.loads((a.guard/'manifest.json').read_text())['images'];local={r['id']:r['boxes'] for r in json.loads((a.guard/'regions/boxes.json').read_text()) if r['model']=='native-yunet-tiled-interior'}
    if len(rows)!=25 or len(local)!=25 or any(r['role']!='dev' for r in rows):raise ValueError('original25 DEV only')
    a.output.mkdir(parents=True,exist_ok=False);inputs=a.output/'inputs';inputs.mkdir();images=[];heads={};omissions=[]
    for source in rows:
        row=dict(source,id=source['source_id']);rawpath=Path(row['rgb']);raw=rawpath.read_bytes();w,h=row['width'],row['height']
        if sha(rawpath)!=row['rgb_sha256'] or len(raw)!=w*h*3:raise ValueError('source RGB differs')
        factor=max(128/min(w,h),512/max(w,h));dw,dh=round(w*factor),round(h*factor)
        if dw*dh>1024**2:raise ValueError('scan input bound exceeded')
        config=f'0,0,{w},{h},448,448;0,0,{w},{h},{dw},{dh}';resized=subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),config],input=raw);offset=448*448*3
        if len(resized)!=offset+dw*dh*3:raise ValueError('owned resize length differs')
        for kind,value,rw,rh in [('gaze',resized[:offset],448,448),('scan',resized[offset:],dw,dh)]:
            target=inputs/(source['id']+'-'+kind+'.rgb8');target.write_bytes(value);row[kind]=dict(path=str(target),width=rw,height=rh,sha256=sha(target))
        proposals=[]
        for x0,y0,x1,y1 in local[row['id']]:
            bw,bh=x1-x0,y1-y0;proposals.append([max(0.,x0-.15*bw),max(0.,y0-.25*bh),min(1.,x1+.15*bw),min(1.,y1+.25*bh)])
        proposals=sorted(proposals,key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),*b));heads[row['id']]={k:row[k] for k in ['role','family_id','rgb_sha256']};heads[row['id']]['people']=[dict(head=b,proposal_confidence_available=False) for b in proposals[:16]];omissions.append(dict(id=row['id'],input_proposals=len(proposals),retained=min(16,len(proposals)),omitted=max(0,len(proposals)-16)));images.append(row)
    (a.output/'heads.json').write_text(json.dumps(heads,indent=2)+'\n');(a.output/'inputs.json').write_text(json.dumps(dict(images=images,teacher_members=teachers,parent_packets={str(a.guard):sha(a.guard/'SHA256SUMS.json'),str(a.attention):sha(a.attention/'SHA256SUMS.json')},resizer_sha256=sha(a.resizer),head_boxes_sha256=sha(a.output/'heads.json'),omissions=omissions,api_calls=0,test_inference=False,new_gradients=False,model_promoted=False),indent=2)+'\n');print(json.dumps(dict(images=25,head_sources=sum(bool(v['people']) for v in heads.values()),heads=sum(len(v['people']) for v in heads.values()),omitted=sum(v['omitted'] for v in omissions),input_bytes=sum(v.stat().st_size for v in inputs.iterdir()))))


if __name__=='__main__':main()
