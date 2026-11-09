#!/usr/bin/env python3
"""Bind fixed real72 teacher inputs to verified owned pixels and existing weights."""
import argparse
import json
from pathlib import Path
import subprocess
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['native','tiles','first_gate_packet','teachers','resizer','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    expected={a.native:'20adfd7777f535cce3ad328992b1f56ae76f7b0b87048f9a299bcdc98789441f',a.tiles:'c7872a992e607cb062089fdd678a0629a16d9f86841c831a4020cd9b8a579b29',
        a.first_gate_packet:'d19651ba88ace98b07175459520b683879cc9bea73db657b490908a67edb5c76'}
    for root,digest in expected.items():
        if sha(root/'SHA256SUMS.json')!=digest:raise ValueError('fixed parent packet differs')
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':raise ValueError('owned resizer differs')
    members=json.loads((a.first_gate_packet/'SHA256SUMS.json').read_text());verified={}
    for key,digest in members.items():
        if key.startswith('first-gate/teachers/'):
            path=a.teachers/key.removeprefix('first-gate/teachers/')
            if sha(path)!=digest:raise ValueError('retained teacher subtree differs: '+key)
            verified[str(path)]=digest
    if len(verified)!=226:raise ValueError('complete pinned teacher subtree required')
    original=json.loads((a.native/'manifest.json').read_text());rows=original['images'];local={r['id']:r['boxes'] for r in json.loads((a.tiles/'analysis/boxes.json').read_text()) if r['model']=='native-yunet-tiled-interior'}
    if len(rows)!=72 or len(local)!=72 or len({r['family_id'] for r in rows})!=72 or sum(r['role']=='fit' for r in rows)!=48 or any(r['role'] not in ['fit','dev'] for r in rows):raise ValueError('fixed independent72 families and48/24 roles required')
    a.output.mkdir(parents=True,exist_ok=False);inputs=a.output/'inputs';inputs.mkdir();images=[];heads={};omissions=[]
    for source in rows:
        row=dict(source,id=source['source_id']);raw=Path(row['rgb']).read_bytes();w,h=row['width'],row['height']
        if sha(Path(row['rgb']))!=row['rgb_sha256'] or len(raw)!=w*h*3:raise ValueError('owned source bytes differ')
        scale=max(128/min(w,h),512/max(w,h));dw,dh=round(w*scale),round(h*scale)
        if dw*dh>1024**2:raise ValueError('one-megapixel scan bound exceeded')
        config=f'0,0,{w},{h},448,448;0,0,{w},{h},{dw},{dh}'
        resized=subprocess.run([str(a.resizer.resolve()),str(w),str(h),str(w*3),config],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
        offset=448*448*3
        if len(resized)!=offset+dw*dh*3:raise ValueError('owned resized output length differs')
        for kind,value,rw,rh in [('gaze',resized[:offset],448,448),('scan',resized[offset:],dw,dh)]:
            target=inputs/(source['id']+'-'+kind+'.rgb8');target.write_bytes(value);row[kind]=dict(path=str(target),width=rw,height=rh,sha256=sha(target))
        proposals=[]
        for x0,y0,x1,y1 in local[row['id']]:
            bw,bh=x1-x0,y1-y0;proposals.append([max(0.,x0-.15*bw),max(0.,y0-.25*bh),min(1.,x1+.15*bw),min(1.,y1+.25*bh)])
        proposals=sorted(proposals,key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),*b))
        heads[row['id']]={k:row[k] for k in ['role','family_id','rgb_sha256']}
        heads[row['id']]['people']=[dict(head=b,proposal_confidence_available=False) for b in proposals[:16]]
        omissions.append(dict(id=row['id'],role=row['role'],input_proposals=len(proposals),retained=min(16,len(proposals)),omitted=max(0,len(proposals)-16)))
        images.append(row)
    (a.output/'heads.json').write_text(json.dumps(heads,indent=2)+'\n')
    protocol=dict(images=images,parent_packets={str(root):digest for root,digest in expected.items()},teacher_members=verified,
        resizer_sha256=sha(a.resizer),head_boxes_sha256=sha(a.output/'heads.json'),omissions=omissions,
        preprocessing='owned Lanczos;448-square gaze/ImageNet; aspect scan min128/max>=512 RGB0..255; transient internal arrays only',
        source_roles_changed=False,api_calls=0,test_inference=False,model_promoted=False,face_coverage_certified=False)
    (a.output/'inputs.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(sources=72,head_sources=sum(bool(v['people']) for v in heads.values()),heads=sum(len(v['people']) for v in heads.values()),omitted_heads=sum(r['omitted'] for r in omissions),input_bytes=sum(p.stat().st_size for p in inputs.iterdir()))))


if __name__=='__main__':main()
