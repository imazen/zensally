#!/usr/bin/env python3
"""Replay old measured attention columns before mixing the broader feature build."""
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
    for name in ['old_cache','old_root','pose','new_cache','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    fixed=[(a.old_cache,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.old_root/'SHA256SUMS.json','86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3'),(a.pose,'b994c0cc2492f5ea412174dcdbd5241c78b2b63e1d8d154d19739b231a8d4064'),(a.new_cache,'863bc60d3e425fb7e84719a282f4287b66d7ab74c2bff052b48132c45fa0941d')]
    if any(sha(path)!=digest for path,digest in fixed):raise ValueError('attention compatibility inputs differ')
    c=torch.load(a.old_cache,map_location='cpu',weights_only=False);new=torch.load(a.new_cache,map_location='cpu',weights_only=False);members=json.loads((a.old_root/'SHA256SUMS.json').read_text());poses={r['id']:r['result'] for r in json.loads(a.pose.read_text())['rows']};documents=[]
    if list(c['feature_names'])!=list(NAMES) or list(new['feature_names'])!=list(NAMES):raise ValueError('32-column feature order differs')
    for source in c['sources']:
        parts=[poses[source['id']]]
        for kind in ['gaze','scan']:
            relative=kind+'/'+source['id']+'.json';path=a.old_root/relative;expected=members[relative] if isinstance(members,dict) else next(r['sha256'] for r in members if r['path']==relative)
            if sha(path)!=expected:raise ValueError('old teacher result changed')
            doc=json.loads(path.read_text())
            if any(doc[k]!=source[k] for k in ['id','role','family_id','rgb_sha256']):raise ValueError('old teacher source identity differs')
            parts.append(doc['result'])
        documents.append(parts)
    counts=0
    for i,source in enumerate(c['sources']):
        mask=c['eligible'][i].flatten();boxes=c['query_boxes'][i].reshape(-1,4)[mask].numpy();j=int(c['donors'][i])
        for key,doc in [('extras',documents[i]),('permuted_extras',documents[j])]:
            actual=torch.from_numpy(candidate_features(boxes,*doc));expected=c[key][i].reshape(-1,32)[mask]
            if not torch.equal(actual,expected):raise ValueError('existing eligible attention feature bits changed')
            counts+=actual.numel()
    result=dict(sources=72,columns=32,real_and_permuted=True,scalar_values_checked=counts,all_bits_exact=True,feature_definition_sha256=sha(Path(__file__).resolve().parents[2]/'training/composition_features.py'),inputs={str(path):digest for path,digest in fixed},scope='existing teacher outputs through unchanged feature function; no new inference',api_calls=0,test_inference=False)
    write(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))


if __name__=='__main__':main()
