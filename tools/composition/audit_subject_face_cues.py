#!/usr/bin/env python3
"""Audit local face/pose proposal scope against unreviewed cached head centers."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys
from artifact_identity import sha
from pose_head_bounds import head_bounds
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime-eval'))
from face_audit import matched


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    doc=json.loads((a.root/'manifest.json').read_text());images={r['id']:r for r in doc['images']}
    completed=json.loads((a.root/'local/summary.json').read_text())
    if len(completed)!=4 or any(r['returncode'] for r in completed):raise ValueError('all four local runs must complete')
    panels={}
    for path in (a.run/'sol-real72').glob('openimages-*.json'):
        result=json.loads(path.read_text())
        if result['annotation_valid']:panels['openimages:'+path.stem.removeprefix('openimages-')]=json.loads((Path(result['entry'])/'zones.json').read_text())
    if len(panels)!=67:raise ValueError('fixed67 valid real Sol annotations required')
    scopes=[];records=[];reasons=Counter();bounds=[]
    for model in doc['models']:
        path=a.root/'local'/model['id']/'results.json';result=json.loads(path.read_text());local={r['id']:r['result'] for r in result['rows']}
        if set(local)!=set(images):raise ValueError('local source set differs')
        returned=0;positive=0
        for sid,value in local.items():
            source=images[sid];w,h=source['width'],source['height'];boxes=[]
            if result['kind']=='faces':
                for detection in value['detections']:
                    x,y,cw,ch=detection['xywh'];boxes.append([x/w,y/h,(x+cw)/w,(y+ch)/h])
            elif result['kind']=='landmarks':
                for points in value['landmarks']:
                    boxes.append([min(p['x'] for p in points),min(p['y'] for p in points),max(p['x'] for p in points),max(p['y'] for p in points)])
            elif result['kind']=='pose':
                for points in value['landmarks']:
                    projected=[dict(p,x=p['x']*w,y=p['y']*h) for p in points]
                    box,reason=head_bounds(projected,w,h);reasons[reason]+=1
                    if box is not None:boxes.append([v/extent for v,extent in zip(box,[w,h,w,h])])
            else:raise ValueError('only local face or pose outputs admitted')
            clipped=[]
            for box in boxes:
                if not all(math.isfinite(v) for v in box):raise ValueError('nonfinite local proposal')
                native=[max(0.,min(1.,float(v))) for v in box]
                if not(native[0]<native[2] and native[1]<native[3]):raise ValueError('invalid local proposal extent')
                clipped.append(native)
            returned+=len(clipped);positive+=bool(clipped)
            bounds.append(dict(id=sid,role=source['role'],model=model['id'],boxes=clipped))
            if sid not in panels:continue
            rois=[r['box'] for r in panels[sid]['rois'] if r['kind']=='face_head'];centers=[[(r['left']+r['right'])/200,(r['top']+r['bottom'])/200] for r in rois]
            hits=matched(centers,clipped)
            records.append(dict(id=sid,role=source['role'],model=model['id'],nominated_heads=len(centers),local_proposals=len(clipped),matched_proposal_centers=len(hits)))
        scopes.append(dict(model=model['id'],kind=result['kind'],sources=len(images),positive_sources=positive,proposals=returned,result_sha256=sha(path),weight_bytes=model['bytes']))
    summary=[]
    for model in doc['models']:
        for role in ['fit','dev']:
            rows=[r for r in records if r['model']==model['id'] and r['role']==role]
            summary.append(dict(model=model['id'],role=role,sources=len(rows),nominated_heads=sum(r['nominated_heads'] for r in rows),
                local_proposals=sum(r['local_proposals'] for r in rows),matched_proposal_centers=sum(r['matched_proposal_centers'] for r in rows)))
    output=a.root/'analysis';output.mkdir(exist_ok=False)
    result=dict(local_scopes=scopes,comparison=summary,pose_head_filter_reasons=dict(reasons),records=records,
        annotation_scope='Sol face/head proposals include backs and occlusions; center inclusion is an annotation-scope diagnostic, not actual-face recall, precision or head containment.',
        face_coverage_certified=False,model_promoted=False,source_roles_changed=False,test_inference=False,api_calls=0)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(output/'boxes.json').write_text(json.dumps(bounds,indent=2)+'\n')
    if sum(p.stat().st_size for p in a.root.rglob('*') if p.is_file())>doc['storage_allowance_bytes']:raise ValueError('registered128MiB cue storage allowance exceeded')
    print(json.dumps(dict(local_scopes=scopes,comparison=summary,pose_head_filter_reasons=dict(reasons))))


if __name__=='__main__':main()
