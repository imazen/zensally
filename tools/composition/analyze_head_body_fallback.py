#!/usr/bin/env python3
"""Separate measured-pose cut changes from changes in offered crop membership."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha
from distill_attention_linear import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('controls',type=Path);p.add_argument('output',type=Path);a=p.parse_args();rows=[];inputs={}
    for parent in ['geometry','person-all-pca8','person-largest-pca8','person-all-raw','person-largest-raw']:
        arms={}
        for policy in ['face-body','face-pose-body']:
            path=a.controls/(parent+'-'+policy)/'predictions.json';inputs[str(path)]=sha(path);values=json.loads(path.read_text());arms[policy]={(r['id'],r['query']):r for r in values}
            if len(arms[policy])!=1584:raise ValueError('all endpoint keys required')
        face,pose=arms.values()
        if set(face)!=set(pose):raise ValueError('source/query membership differs')
        keys=[key for key,r in face.items() if r['role']=='dev' and r['query']<18 and r['head_known'] and r['reference_known']]
        if len(keys)!=139:raise ValueError('fixed accepted DEV denominator differs')
        common=[key for key in keys if face[key]['offered'] and pose[key]['offered']]
        rows.append(dict(parent=parent,accepted=139,common_offered=len(common),face_only_cuts_on_common=sum(face[k]['weak_cut'] for k in common),face_pose_cuts_on_common=sum(pose[k]['weak_cut'] for k in common),common_cut_repairs=sum(face[k]['weak_cut'] and not pose[k]['weak_cut'] for k in common),common_new_cuts=sum(not face[k]['weak_cut'] and pose[k]['weak_cut'] for k in common),newly_withheld=sum(face[k]['offered'] and not pose[k]['offered'] for k in keys),newly_offered=sum(not face[k]['offered'] and pose[k]['offered'] for k in keys),common_reference_gains=sum(not face[k]['reference_match'] and pose[k]['reference_match'] for k in common),common_reference_losses=sum(face[k]['reference_match'] and not pose[k]['reference_match'] for k in common)))
    result=dict(rows=rows,inputs=inputs,scope='reused DEV mechanism diagnostic; not independent quality confirmation',api_calls=0,test_inference=False,model_promoted=False);write(a.output,result);print(json.dumps(rows))


if __name__=='__main__':main()
