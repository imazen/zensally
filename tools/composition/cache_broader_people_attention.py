#!/usr/bin/env python3
"""Actual pose/gaze/scan summaries at broader fixed candidate geometry."""
import argparse
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from distill_attention_linear import write
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from composition_features import candidate_features,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','regions','attention','cues','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='b027605dc44482458ba939eae54ab0035f95599701709b31c78f36cb9cfe7dd8' or sha(a.regions)!='72346b4a8ec6875eb047aa393b177c6aeb3c665b2eb49a0678e6310c1df4523a':raise ValueError('fixed broader cache/proposals differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);regions=json.loads(a.regions.read_text());poses={r['id']:r['result'] for r in json.loads((a.cues/'mediapipe/local/float16-pose_landmarker_lite/results.json').read_text())['rows']};documents=[];identities={str(a.cache):sha(a.cache),str(a.regions):sha(a.regions)}
    for kind in ['gaze','scan']:
        summary=json.loads((a.attention/kind/'summary.json').read_text())
        if summary['completed']!=16 or summary['required']!=16 or summary['diagnostic_maps_saved']:raise ValueError('all16 compact teacher outputs required')
        identities[str(a.attention/kind/'summary.json')]=sha(a.attention/kind/'summary.json')
    for source in c['sources']:
        values=[poses[source['id']]]
        for kind in ['gaze','scan']:
            path=a.attention/kind/(source['id']+'.json');doc=json.loads(path.read_text())
            if any(doc[k]!=source[k] for k in ['id','role','family_id','rgb_sha256']):raise ValueError('teacher source identity differs')
            values.append(doc['result']);identities[str(path)]=sha(path)
        documents.append(values)
    real=[];permuted=[]
    for i,source in enumerate(c['sources']):
        j=regions['donors'][i]
        if c['sources'][j]['role']!=source['role']:raise ValueError('donor crosses roles')
        boxes=c['query_boxes'][i].reshape(-1,4).numpy();real.append(torch.from_numpy(candidate_features(boxes,*documents[i])).reshape(22,128,32));permuted.append(torch.from_numpy(candidate_features(boxes,*documents[j])).reshape(22,128,32))
    result=dict(schema='broader-candidate-attention-v1',sources=c['sources'],extras=torch.stack(real),permuted_extras=torch.stack(permuted),donors=regions['donors'],feature_names=NAMES,identities=identities,api_calls=0,test_inference=False)
    if not torch.isfinite(result['extras']).all() or not torch.isfinite(result['permuted_extras']).all():raise ValueError('nonfinite candidate attention features')
    a.output.mkdir(parents=True,exist_ok=False);torch.save(result,a.output/'cache.pt');write(a.output/'summary.json',dict(sources=16,shape=list(result['extras'].shape),cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,pose_positive=sum(bool(v[0]['landmarks']) for v in documents),gaze_positive=sum(bool(v[1]['people']) for v in documents),api_calls=0,test_inference=False));print(json.dumps(json.loads((a.output/'summary.json').read_text())))


if __name__=='__main__':main()
