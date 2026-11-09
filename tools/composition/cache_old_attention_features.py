#!/usr/bin/env python3
"""Cache measured original DEV attention features without changing frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from composition_features import candidate_features,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','guard','rank']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.rank)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.guard/'SHA256SUMS.json')!='0c1909193ec1b219adb22a005cfbd1e32d09929e6890ef0d6e8820c28b8f54e5':raise ValueError('frozen original inputs differ')
    members=json.loads((a.guard/'SHA256SUMS.json').read_text());pose_path=a.guard/'mediapipe/float16-pose_landmarker_lite/results.json'
    if sha(pose_path)!=members[str(pose_path.relative_to(a.guard))]:raise ValueError('pinned pose differs')
    poses={r['id']:r['result'] for r in json.loads(pose_path.read_text())['rows']};inputs=json.loads((a.root/'inputs.json').read_text());lookup={r['id']:r for r in inputs['images']};rank=torch.load(a.rank,map_location='cpu',weights_only=False);indices=[i for i,s in enumerate(rank['sources']) if s['role']=='dev'];sources=[rank['sources'][i] for i in indices]
    if len(sources)!=25 or set(lookup)!={s['id'] for s in sources}:raise ValueError('original25 DEV only')
    for kind in ['gaze','scan']:
        summary=json.loads((a.root/kind/'summary.json').read_text())
        if summary['completed']!=25 or summary['required']!=25 or summary['diagnostic_maps_saved']:raise ValueError('complete compact teacher output required')
    documents=[]
    for source in sources:
        row=lookup[source['id']]
        if any(row[k]!=source[k] for k in ['role','family_id','rgb_sha256']):raise ValueError('teacher source identity differs')
        parts=[]
        for kind in ['gaze','scan']:
            doc=json.loads((a.root/kind/(source['id']+'.json')).read_text())
            if any(doc[k]!=source[k] for k in ['id','role','family_id','rgb_sha256']):raise ValueError('compact teacher output identity differs')
            parts.append(doc['result'])
        documents.append((poses[row['id'].replace(':','-',1)],*parts))
    groups={};donors=list(range(25))
    for j,i in enumerate(indices):groups.setdefault(tuple((rank['content'][i]*256).round().int().tolist()),[]).append(j)
    for values in groups.values():
        ordered=sorted(values,key=lambda j:hashlib.sha256(('direct-crop-permutation|'+sources[j]['id']).encode()).hexdigest())
        for i,j in zip(ordered,ordered[1:]+ordered[:1]):donors[i]=j
    real=[];permuted=[]
    for j,i in enumerate(indices):
        boxes=rank['boxes'][i].repeat_interleave(2,0).reshape(-1,4);valid=rank['eligible'][i].flatten();r=torch.zeros(22*128,32);p=torch.zeros_like(r);geometry=boxes[valid].numpy();r[valid]=torch.from_numpy(candidate_features(geometry,*documents[j]));p[valid]=torch.from_numpy(candidate_features(geometry,*documents[donors[j]]));real.append(r.reshape(22,128,32));permuted.append(p.reshape(22,128,32))
    result=dict(source_ids=[s['id'] for s in sources],sources=sources,indices=indices,extras=torch.stack(real),permuted_extras=torch.stack(permuted),donors=donors,names=NAMES,eligible=rank['eligible'][indices],rank_sha256=sha(a.rank),input_identity_sha256=sha(a.root/'inputs.json'),pose_sha256=sha(pose_path),api_calls=0,test_inference=False,new_gradients=False)
    if not torch.isfinite(result['extras']).all() or not torch.isfinite(result['permuted_extras']).all():raise ValueError('nonfinite measured features')
    path=a.root/'candidate-features.pt'
    if path.exists():raise ValueError('refuse to overwrite measured features')
    torch.save(result,path);summary=dict(cache_sha256=sha(path),cache_bytes=path.stat().st_size,sources=25,pose_sources=sum(bool(d[0]['landmarks']) for d in documents),gaze_sources=sum(bool(d[1]['people']) for d in documents),donors_moved=sum(i!=j for i,j in enumerate(donors)),eligible_slots=int(result['eligible'].sum()),unused_slots_explicit_masked_placeholders=True,api_calls=0,test_inference=False,new_gradients=False,model_promoted=False);(a.root/'feature-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
