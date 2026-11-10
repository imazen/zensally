#!/usr/bin/env python3
"""Project real and fixed donor head relationships at recipient crop boxes."""
import argparse
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from relationship_features import features,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','attention','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed parent cache differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);docs=[];identities={}
    for source in cache['sources']:
        path=a.attention/'gaze'/(source['id']+'.json');doc=json.loads(path.read_text())
        if any(doc[k]!=source[k] for k in ['id','family_id','role']):raise ValueError('source/family/role mismatch')
        docs.append(doc['result']['people']);identities[str(path)]=sha(path)
    real=[];permuted=[]
    for i,source in enumerate(cache['sources']):
        donor=int(cache['donors'][i])
        if source['role']!=cache['sources'][donor]['role']:raise ValueError('donor crosses roles')
        boxes=cache['query_boxes'][i].reshape(-1,4).numpy()
        real.append(torch.from_numpy(features(boxes,docs[i])).reshape(22,128,16))
        permuted.append(torch.from_numpy(features(boxes,docs[donor])).reshape(22,128,16))
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(real=torch.stack(real),permuted=torch.stack(permuted),parent_cache_sha256=sha(a.cache),source_ids=[s['id'] for s in cache['sources']],feature_names=NAMES,teacher_identities=identities),a.output/'features.pt')
    (a.output/'summary.json').write_text(json.dumps(dict(features_sha256=sha(a.output/'features.pt'),sources=72,head_sources=sum(bool(p) for p in docs),multihead_sources=sum(len(p)>1 for p in docs),api_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
