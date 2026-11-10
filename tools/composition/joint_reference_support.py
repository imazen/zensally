#!/usr/bin/env python3
"""Frozen candidate and exact-reference ceilings for nested same-story crops."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from artifact_identity import sha
from refinement_inputs import accepted_options,reference_match
from subject_policy import face_relation
from crop_corridors import audit
from nested_pair_training import compatibility
from distill_attention_linear import write


def signatures(boxes,eligible,heads):
    result=[]
    for box,keep in zip(boxes,eligible):
        if not keep:result.append(None);continue
        relations=[face_relation(box,h) for h in heads]
        result.append(None if 'cut' in relations else sum(1<<i for i,v in enumerate(relations) if v=='contained'))
    return result


def interval_mask(outer,inner,eligible,heads):
    left=signatures(outer,eligible[0],heads);right=signatures(inner,eligible[1],heads);same=np.equal.outer(np.asarray(left,dtype=object),np.asarray(right,dtype=object));valid=np.asarray([v is not None for v in left])[:,None]&np.asarray([v is not None for v in right])[None,:];retained=np.asarray([v is not None and v!=0 for v in left])[:,None]
    return same&valid,same&valid&retained


def bank_compatibility(outer,inner,eligible):
    n=len(outer);boxes=torch.tensor([*outer,*inner],dtype=torch.int64);e=torch.zeros(2,n*2,dtype=torch.bool);e[0,:n]=torch.as_tensor(eligible[0]);e[1,n:]=torch.as_tensor(eligible[1]);return compatibility(boxes,e).numpy()[:n,n:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','references','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('frozen source/label cache differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False)
    if len(c['sources'])!=72 or any(s['role'] not in ['fit','dev'] for s in c['sources']):raise ValueError('fixed FIT/DEV only; TEST closed')
    if {s['family_id'] for s in c['sources'] if s['role']=='fit'}&{s['family_id'] for s in c['sources'] if s['role']=='dev'}:raise ValueError('family crosses role')
    scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().long();options,heads,ids,replayed=accepted_options(dict(data=c,native=native),a.references);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(cache_sha256=sha(a.cache),references=ids,positive_mask_replayed_queries=replayed,source_roles_unchanged=True,scope='finite-bank and accepted exact-option oracle diagnostics, no new labels',api_calls=0,test_inference=False,model_inference=False,new_gradients=False,model_promoted=False));rows=[];status=dict(pid=os.getpid(),stage='joint-support',completed=0,required=72)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure)
    for i,s in enumerate(c['sources']):
        sid=s['id'];w,h=s['native_size'];known=s['annotation_valid'] and bool(heads[sid]);progress(source=sid,completed=i)
        for ri,ratio in enumerate(c['ratios']):
            q=ri*2;b=native[i,q:q+2].tolist();e=c['eligible'][i,q:q+2].numpy();positive=c['positive'][i,q:q+2].numpy();ref_known=c['reference_valid'][i,q:q+2].any(-1).numpy();mask=bank_compatibility(b[0],b[1],e);safe,retained=interval_mask(b[0],b[1],e,heads[sid]);safe&=mask;retained&=mask;known_score=(positive[0][:,None]&ref_known[0]).astype(int)+(positive[1][None,:]&ref_known[1]).astype(int);stories=[]
            for m in range(2):
                grouped={}
                for option in options.get((sid,q+m),[]):
                    story=tuple(sorted(option['args'][1]));flags=np.array([reference_match(box,[option],w,h) if keep else False for box,keep in zip(b[m],positive[m])]);grouped[story]=grouped.get(story,np.zeros(128,bool))|flags
                stories.append(grouped)
                union=np.logical_or.reduce(list(grouped.values())) if grouped else np.zeros(128,bool)
                if not np.array_equal(union,positive[m]):raise ValueError('story masks differ from frozen accepted positives')
            def bound(which):return int(known_score[which].max()) if which.any() else None
            def joint(which):return bool((which&positive[0][:,None]&positive[1][None,:]).any())
            same_story=any((safe&stories[0][story][:,None]&stories[1][story][None,:]).any() for story in stories[0].keys()&stories[1].keys());exact=[]
            for oi,outer in enumerate(options.get((sid,q),[])):
                for ii,inner in enumerate(options.get((sid,q+1),[])):
                    ob=np.rint(outer['target']*[w,h,w,h]).astype(int).tolist();ib=np.rint(inner['target']*[w,h,w,h]).astype(int).tolist();assessment=audit(ib,ob,w,h,ratio,heads[sid]);strict=(ib[2]-ib[0])*(ib[3]-ib[1])<(ob[2]-ob[0])*(ob[3]-ob[1]);same=sorted(outer['args'][1])==sorted(inner['args'][1]);exact.append(dict(outer_option=oi,inner_option=ii,outer=ob,inner=ib,outer_subjects=outer['args'][1],inner_subjects=inner['args'][1],same_story=same,strict_nested=assessment['nested'] and assessment['exact_aspect'] and strict,interval_safe=assessment['geometrically_admissible'] and strict))
            rows.append(dict(id=sid,role=s['role'],family_id=s['family_id'],source=s['source'],ratio=ratio,ratio_index=ri,head_known=known,reference_known=ref_known.tolist(),known_endpoints=int(ref_known.sum()),positive_slots=positive.sum(-1).tolist(),independent_supported_endpoints=int((positive.any(-1)&ref_known).sum()),bank_strict_bound=bound(mask),bank_interval_bound=bound(safe),bank_retained_head_bound=bound(retained),bank_strict_pair=joint(mask),bank_interval_pair=joint(safe),bank_same_story_interval_pair=same_story,exact_options=exact,exact_strict_pair=any(v['strict_nested'] for v in exact),exact_interval_pair=any(v['interval_safe'] for v in exact),exact_same_story_interval_pair=any(v['interval_safe'] and v['same_story'] for v in exact)))
        progress(completed=i+1)
    groups=[]
    for role in ['fit','dev']:
        for secondary in [False,True]:
            selected=[r for r in rows if r['role']==role and (r['ratio_index']>=9)==secondary and r['head_known']];both=[r for r in selected if all(r['reference_known'])];one=[r for r in selected if r['known_endpoints']];groups.append(dict(role=role,secondary=secondary,known_ratio_pairs=len(selected),accepted_endpoints=sum(r['known_endpoints'] for r in selected),independent_supported_endpoints=sum(r['independent_supported_endpoints'] for r in selected),strict_bank_match_ceiling=sum(r['bank_strict_bound'] or 0 for r in one),interval_bank_match_ceiling=sum(r['bank_interval_bound'] or 0 for r in one),retained_head_bank_match_ceiling=sum(r['bank_retained_head_bound'] or 0 for r in one),accepted_pairs=len(one),no_strict_pair=sum(r['bank_strict_bound'] is None for r in one),no_interval_pair=sum(r['bank_interval_bound'] is None for r in one),both_reference_pairs=len(both),**{key:sum(r[key] for r in both) for key in ['bank_strict_pair','bank_interval_pair','bank_same_story_interval_pair','exact_strict_pair','exact_interval_pair','exact_same_story_interval_pair']}))
    write(a.output/'summary.json',dict(groups=groups,records=len(rows),api_calls=0,test_inference=False,new_gradients=False,model_promoted=False));write(a.output/'sources.json',c['sources'])
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    progress(pid=None,stage='complete',completed=72);print(json.dumps(groups))


if __name__=='__main__':main()
