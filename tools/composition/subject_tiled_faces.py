#!/usr/bin/env python3
"""Bounded four-tile YuNet coverage control on the fixed real72 panel."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from artifact_identity import sha
from subject_native_faces import BINARY_SHA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime-eval'))
from face_audit import matched


def windows(width,height):
    if type(width) is not int or type(height) is not int or min(width,height)<2:raise ValueError('positive source dimensions required')
    tw,th=(width*3+4)//5,(height*3+4)//5
    return [[x,y,x+tw,y+th] for y in [0,height-th] for x in [0,width-tw]]


def extract(rgb,width,height,box):
    if len(rgb)!=width*height*3 or not(0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('packed owned RGB and bounded tile required')
    return b''.join(rgb[(y*width+box[0])*3:(y*width+box[2])*3] for y in range(box[1],box[3]))


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def merge(rows):
    output=[]
    for row in sorted(rows,key=lambda r:(-r['confidence'],r['box'],r['origin'])):
        if all(iou(row['box'],other['box'])<=.3 for other in output):output.append(row)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','audit'])
    for key in ['native','run','binary','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    if sha(a.binary)!=BINARY_SHA or sha(a.native/'SHA256SUMS.json')!='20adfd7777f535cce3ad328992b1f56ae76f7b0b87048f9a299bcdc98789441f':raise ValueError('fixed native inputs differ')
    source=json.loads((a.native/'manifest.json').read_text());sources=source['images']
    if len(sources)!=72 or any(r['role'] not in ['fit','dev'] for r in sources):raise ValueError('fixed72 FIT/DEV source panel required')
    if a.action=='prepare':
        a.output.mkdir(parents=True,exist_ok=False);(a.output/'rgb').mkdir();images=[]
        for row in sources:
            rgb=Path(row['rgb']).read_bytes()
            if sha(Path(row['rgb']))!=row['rgb_sha256']:raise ValueError('owned source pixels differ')
            for index,box in enumerate(windows(row['width'],row['height'])):
                name=row['id']+'-tile'+str(index);path=a.output/'rgb'/(name+'.rgb')
                path.write_bytes(extract(rgb,row['width'],row['height'],box))
                images.append(dict(id=name,source_id=row['source_id'],role=row['role'],rgb=str(path),rgb_sha256=sha(path),
                    width=box[2]-box[0],height=box[3]-box[1],source_box=box,source_size=[row['width'],row['height']]))
        if len(images)!=288:raise ValueError('exactly four fixed tiles per source required')
        (a.output/'manifest.json').write_text(json.dumps(dict(images=images,models=[],requested_native_models=['yunet'],
            native_binary_sha256=BINARY_SHA,native_packet_sha256=sha(a.native/'SHA256SUMS.json'),
            window_rule='four corner windows of ceil(3/5 width) by ceil(3/5 height); no resize before retained probe',
            policies=['full-source cached','full plus four tiles, NMS IoU0.3','same with2-source-pixel internal-tile-edge rejection'],
            source_roles_changed=False,api_calls=0,test_inference=False,face_coverage_certified=False,model_promoted=False),indent=2)+'\n')
        (a.output/'inputs.tsv').write_text(''.join(f"{r['id']}\t{r['width']}\t{r['height']}\t{r['rgb']}\t0\n" for r in images))
        print(json.dumps(dict(sources=72,tiles=288,rgb_bytes=sum(Path(r['rgb']).stat().st_size for r in images))))
        return
    complete=json.loads((a.output/'local/summary.json').read_text())
    if len(complete)!=1 or complete[0]['model']!='yunet' or complete[0]['returncode']:raise ValueError('native tile evaluation must complete')
    manifest=json.loads((a.output/'manifest.json').read_text());tiles={r['id']:r for r in manifest['images']}
    rows=json.loads((a.output/'local/yunet/results.json').read_text())
    if len(rows)!=288 or {r['id'] for r in rows}!=set(tiles):raise ValueError('native tile output set differs')
    baseline=json.loads((a.native/'local/yunet/results.json').read_text());proposals={}
    for row in baseline:
        sid=next(r['source_id'] for r in sources if r['id']==row['id'])
        proposals[sid]=[dict(box=[max(0.,min(1.,v/100)) for v in face['box_percent']],confidence=face['confidence'],origin='full',internal_edge=False) for face in row['result']['faces']]
    raw=[]
    for row in rows:
        tile=tiles[row['id']];x,y,right,bottom=tile['source_box'];w,h=tile['source_size'];tw,th=tile['width'],tile['height']
        if sha(Path(tile['rgb']))!=tile['rgb_sha256']:raise ValueError('retained tile pixels differ')
        for face in row['result']['faces']:
            if not all(math.isfinite(v) for v in face['box_percent']+[face['confidence']]):raise ValueError('nonfinite raw tile result')
            box=[max(0.,min(1.,v/100)) for v in face['box_percent']]
            if not all(math.isfinite(v) for v in box+[face['confidence']]) or not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('invalid tile face geometry')
            mapped=[(x+box[0]*tw)/w,(y+box[1]*th)/h,(x+box[2]*tw)/w,(y+box[3]*th)/h]
            edge=(x>0 and box[0]*tw<=2) or (y>0 and box[1]*th<=2) or (right<w and (1-box[2])*tw<=2) or (bottom<h and (1-box[3])*th<=2)
            item=dict(box=mapped,confidence=face['confidence'],origin=tile['id'],internal_edge=bool(edge));proposals[tile['source_id']].append(item);raw.append(dict(source_id=tile['source_id'],**item))
    boxes=json.loads((a.native/'analysis/boxes.json').read_text());records=[];scopes=[]
    for policy in ['native-yunet-tiled','native-yunet-tiled-interior']:
        count=positive=0
        for row in sources:
            sid=row['source_id'];chosen=merge([r for r in proposals[sid] if policy=='native-yunet-tiled' or not r['internal_edge']]);regions=[r['box'] for r in chosen]
            count+=len(regions);positive+=bool(regions);boxes.append(dict(id=sid,role=row['role'],model=policy,boxes=regions))
            result=json.loads((a.run/'sol-real72'/(sid.replace(':','-')+'.json')).read_text())
            if not result['annotation_valid']:continue
            rois=[r['box'] for r in json.loads((Path(result['entry'])/'zones.json').read_text())['rois'] if r['kind']=='face_head']
            centers=[[(r['left']+r['right'])/200,(r['top']+r['bottom'])/200] for r in rois]
            records.append(dict(id=sid,role=row['role'],model=policy,nominated_heads=len(centers),local_proposals=len(regions),matched_proposal_centers=len(matched(centers,regions))))
        scopes.append(dict(model=policy,sources=72,positive_sources=positive,proposals=count))
    grouped=defaultdict(list)
    for row in records:grouped[(row['model'],row['role'])].append(row)
    comparison=[dict(model=model,role=role,sources=len(rows),**{k:sum(r[k] for r in rows) for k in ['nominated_heads','local_proposals','matched_proposal_centers']}) for (model,role),rows in grouped.items()]
    out=a.output/'analysis';out.mkdir(exist_ok=False)
    (out/'boxes.json').write_text(json.dumps(boxes,indent=2)+'\n');(out/'tile-proposals.json').write_text(json.dumps(raw,indent=2)+'\n')
    summary=dict(local_scopes=scopes,comparison=comparison,records=records,newly_evaluated_calls=288,cached_full_source_calls=72,
        test_inference=False,face_coverage_certified=False,model_promoted=False,api_calls=0,
        limitation='Incomplete unreviewed Sol head-center overlap, not actual face recall or whole-head containment; no runtime speed claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(local_scopes=scopes,comparison=comparison)))


if __name__=='__main__':main()
