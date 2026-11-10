#!/usr/bin/env python3
"""Bind source-only local/oracle region descriptors to the fixed direct-crop cache."""
import argparse
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from region_context import descriptor,FEATURES

VARIANTS=['none','local-face','local-pose','local-face-pose','oracle-heads']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['direct_cache','cues','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.direct_cache)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472' or sha(a.cues/'SHA256SUMS.json')!='d69e2460bfe6c0fd6a680969a35eb0828bb697049c22fb81cb64cfb65755215a':raise ValueError('fixed direct/cue parents differ')
    members={r['path']:r for r in json.loads((a.cues/'SHA256SUMS.json').read_text())}
    if sha(a.cues/'analysis/boxes.json')!=members['analysis/boxes.json']['sha256']:raise ValueError('fixed local boxes differ')
    parent=torch.load(a.direct_cache,map_location='cpu',weights_only=False);sources=parent['sources']
    local={}
    for row in json.loads((a.cues/'analysis/boxes.json').read_text()):local.setdefault(row['id'],{})[row['model']]=row
    if set(local)!={s['id'] for s in sources}:raise ValueError('local/direct source set differs')
    values={k:[] for k in VARIANTS};records=[]
    for i,source in enumerate(sources):
        rows=local[source['id']]
        if len(rows)!=4 or any(r['role']!=source['role'] for r in rows.values()):raise ValueError('fixed local role/model set differs')
        face=rows['float16-blaze_face_full_range_sparse']['boxes'];pose=rows['float16-pose_landmarker_lite']['boxes']
        oracle=[[r[k]/100 for k in ['left','top','right','bottom']] for r in parent['head_rois'][i]]
        for variant,boxes in [('none',[]),('local-face',face),('local-pose',pose),('local-face-pose',face+pose),('oracle-heads',oracle)]:
            vector,record=descriptor(boxes);values[variant].append(vector);records.append(dict(id=source['id'],role=source['role'],variant=variant,**record))
    result=dict(schema='source-region-context-v1',sources=sources,variants={k:torch.tensor(v,dtype=torch.float32) for k,v in values.items()},
        parent_direct_cache_sha256=sha(a.direct_cache),parent_cue_packet_sha256=sha(a.cues/'SHA256SUMS.json'),features=FEATURES,
        protocol=dict(roi_slots=8,global_region_extent_centroid_count=True,only_source_geometry_inputs=True,
            crop_windows_and_selected_subject_ids_are_not_inputs=True,oracle_heads_are_same_call_unreviewed_annotations=True,
            local_face='frozen sparse full-range BlazeFace',local_pose='frozen Pose Lite padded-head estimates',
            source_roles_changed=False,test_inference=False,api_calls=0,face_coverage_certified=False,model_promoted=False))
    a.output.mkdir(parents=True,exist_ok=False);torch.save(result,a.output/'cache.pt')
    (a.output/'regions.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(sources=len(sources),features=FEATURES,variants=VARIANTS,cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,
        omitted_slots={k:sum(r['omitted_slots'] for r in records if r['variant']==k) for k in VARIANTS})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
