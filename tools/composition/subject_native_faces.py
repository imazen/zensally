#!/usr/bin/env python3
"""Evaluate retained native face baselines on fixed real72 pixels and preserve IDs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime-eval'))
from face_audit import matched

BINARY_SHA='7410924fc56b81c1759207af756ccf75034ec374da042067c78baa767b7c9dd4'
CUE_PACKET='d69e2460bfe6c0fd6a680969a35eb0828bb697049c22fb81cb64cfb65755215a'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','audit'])
    for key in ['cues','run','binary','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    if sha(a.binary)!=BINARY_SHA or sha(a.cues/'SHA256SUMS.json')!=CUE_PACKET:raise ValueError('retained native binary/cue inputs differ')
    if a.action=='prepare':
        original=json.loads((a.cues/'manifest.json').read_text());images=[]
        for row in original['images']:
            if row['role'] not in ['fit','dev'] or sha(Path(row['rgb']))!=row['rgb_sha256']:raise ValueError('fixed owned pixels/role differ')
            images.append(dict(row,source_id=row['id'],id=row['id'].replace(':','-')))
        if len(images)!=72 or len({r['id'] for r in images})!=72:raise ValueError('one-to-one fixed72 source IDs required')
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'manifest.json').write_text(json.dumps(dict(images=images,models=[],requested_native_models=['ultraface','yunet'],
            native_binary=str(a.binary),native_binary_sha256=BINARY_SHA,source_cue_packet_sha256=CUE_PACKET,
            source_roles_changed=False,test_inference=False,api_calls=0,diagnostic_maps_requested=False),indent=2)+'\n')
        (a.output/'inputs.tsv').write_text(''.join(f"{r['id']}\t{r['width']}\t{r['height']}\t{r['rgb']}\t0\n" for r in images))
        print(json.dumps(dict(sources=72,models=2,expected_calls=144)))
        return
    manifest=json.loads((a.output/'manifest.json').read_text());completed=json.loads((a.output/'local/summary.json').read_text())
    if {r['model'] for r in completed}!={'ultraface','yunet'} or any(r['returncode'] for r in completed):raise ValueError('both fixed native face evaluations must complete')
    ids={r['id']:r for r in manifest['images']};boxes=json.loads((a.cues/'analysis/boxes.json').read_text());scopes=[];records=[]
    for model in ['ultraface','yunet']:
        path=a.output/'local'/model/'results.json';rows=json.loads(path.read_text())
        if len(rows)!=72 or {r['id'] for r in rows}!=set(ids):raise ValueError('native output ID set differs')
        count=positive=0
        for row in rows:
            source=ids[row['id']];sid=source['source_id'];regions=[]
            for face in row['result']['faces']:
                raw=face['box_percent']
                if not all(math.isfinite(v) for v in raw+[face['confidence']]):raise ValueError('nonfinite native face result')
                region=[max(0.,min(1.,v/100)) for v in raw]
                if not(0<=region[0]<region[2]<=1 and 0<=region[1]<region[3]<=1):raise ValueError('invalid native face geometry')
                regions.append(region)
            count+=len(regions);positive+=bool(regions);boxes.append(dict(id=sid,role=source['role'],model='native-'+model,boxes=regions))
            result=json.loads((a.run/'sol-real72'/(sid.replace(':','-')+'.json')).read_text())
            if not result['annotation_valid']:continue
            rois=[r['box'] for r in json.loads((Path(result['entry'])/'zones.json').read_text())['rois'] if r['kind']=='face_head']
            centers=[[(r['left']+r['right'])/200,(r['top']+r['bottom'])/200] for r in rois]
            records.append(dict(id=sid,role=source['role'],model='native-'+model,nominated_heads=len(centers),local_proposals=len(regions),matched_proposal_centers=len(matched(centers,regions))))
        scopes.append(dict(model='native-'+model,sources=72,positive_sources=positive,proposals=count,result_sha256=sha(path)))
    grouped=defaultdict(list)
    for row in records:grouped[(row['model'],row['role'])].append(row)
    comparison=[dict(model=model,role=role,sources=len(rows),**{k:sum(r[k] for r in rows) for k in ['nominated_heads','local_proposals','matched_proposal_centers']}) for (model,role),rows in grouped.items()]
    output=a.output/'analysis';output.mkdir(exist_ok=False)
    (output/'boxes.json').write_text(json.dumps(boxes,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(local_scopes=scopes,comparison=comparison,records=records,
        newly_evaluated_calls=144,cached_mediapipe_calls=288,native_binary_sha256=BINARY_SHA,source_roles_changed=False,
        face_coverage_certified=False,model_promoted=False,test_inference=False,api_calls=0,
        limitation='Matched Sol head-proposal centers are an annotation-scope diagnostic, not human face recall or complete head containment.'),indent=2)+'\n')
    print(json.dumps(dict(local_scopes=scopes,comparison=comparison)))


if __name__=='__main__':main()
