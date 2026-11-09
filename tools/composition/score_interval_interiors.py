#!/usr/bin/env python3
"""Finite interior rank diagnostics with frozen teacher normalization anchors."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from cached_focalnet import CachedContext
from qualify_offbank_teacher import prepared,bits_equal
from distill_attention_linear import write
from refine_nested_origin import refine  # Establish the shared corridor import path.
from crop_corridors import samples,member


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rank','teacher','study','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.rank)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.study/'SHA256SUMS.json')!='6c8dada6dc31c97770933c1dd50a00ea73586e64baae665adb3dce18bef9b165' or sha(a.teacher/'SHA256SUMS.json')!='a49bbedcd4894ed7a7dd0c5a9f6b0f3a31647aba8f4e02d5675caa95e2e4d04e':raise ValueError('frozen interval inputs differ')
    for name in ['geometry/parents.jsonl','geometry/predictions.jsonl','teacher-qualification/normalization-anchors.json']:verify_member(a.study,name)
    for name in ['analysis.onnx','query.onnx']:verify_member(a.teacher,'runtime/split/'+name)
    c=torch.load(a.rank,map_location='cpu',weights_only=False);indices=[i for i,s in enumerate(c['sources']) if s['role']=='dev'];anchors={(r['id'],r['query']):r for r in json.loads((a.study/'teacher-qualification/normalization-anchors.json').read_text())}
    if len(indices)!=25:raise ValueError('original25 DEV only')
    pairs=[]
    for path,policy in [('geometry/parents.jsonl','parent'),('geometry/predictions.jsonl','refined')]:
        with (a.study/path).open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['cohort']!='old' or (policy=='refined' and r['policy']!='faces-pose/pad25'):continue
                boxes=samples(r['inner'],r['outer'],r['ratio']) if r['inner'] is not None else None
                if boxes is not None and any(not member(box,r['inner'],r['outer'],r['ratio']) for box in boxes):raise ValueError('sample outside exact interval')
                pairs.append(dict(seed=r['seed'],id=r['id'],ratio=r['ratio'],ratio_index=r['ratio_index'],policy=policy,boxes=boxes))
    requests={}
    for row in pairs:
        for box in row['boxes'] or []:requests.setdefault((row['id'],row['ratio_index']),set()).add(tuple(box))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):sha(path) for path in [a.rank,a.teacher/'SHA256SUMS.json',a.study/'SHA256SUMS.json']},analysis_threads=4,query_threads=1,interior_samples=[.25,.5,.75],strong_drop_margin=.25,normalization='frozen original anchors and validity',scope='finite scalar diagnostics only; no excellent-interval or continuous-composition claim',api_calls=0,test_inference=False,new_gradients=False,model_promoted=False));context=CachedContext(a.teacher/'runtime/split',threads=4,query_threads=1);status=dict(pid=os.getpid(),stage='scoring-interiors',completed=0,required=25);raw={};insertions=0;replays=0
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure)
    for number,i in enumerate(indices):
        s=c['sources'][i];sid=s['id'];w,h=s['native_size'];image,bounds=prepared(c,i);token=context.analyze(image,bounds,(sid,s['rgb_sha256']));progress(completed=number,source=sid)
        for ri,ratio in enumerate(c['ratios']):
            boxes=c['boxes'][i:i+1,ri].numpy().copy();reference=c['teacher_scores'][i,ri].numpy();actual=context.score(boxes,token)[0].copy()
            if not bits_equal(actual,reference):raise ValueError('original bank bits changed')
            replays+=1;existing={tuple(box):float(reference[k]) for k,box in enumerate(c['native_boxes'][i][ri])};slot=int(np.flatnonzero((boxes[0]==0).all(-1))[0])
            for m in range(2):
                q=ri*2+m;anchor=anchors[sid,q];eligible=c['eligible'][i,q].numpy()
                if bool(c['query_valid'][i,q])!=anchor['valid']:raise ValueError('frozen query validity differs')
                if anchor['valid'] and not bits_equal((reference[eligible]-np.float32(anchor['original_mean']))/np.float32(anchor['original_std']),c['targets'][i,q].numpy()[eligible]):raise ValueError('frozen normalization target bits differ')
            for box in sorted(requests[sid,ri]):
                if box in existing:value=existing[box]
                else:
                    changed=boxes.copy();changed[0,slot]=np.asarray(box,np.float64)/[w,h,w,h];scores=context.score(changed,token)[0].copy();keep=np.arange(128)!=slot
                    if not bits_equal(scores[keep],reference[keep]):raise ValueError('interior insertion changes original candidates')
                    value=float(scores[slot]);insertions+=1
                raw[sid,ri,box]=value
        progress(completed=number+1)
    records=[]
    for pair in pairs:
        ri=pair['ratio_index'];values=[raw[pair['id'],ri,tuple(box)] for box in pair['boxes']] if pair['boxes'] is not None else None
        for m in range(2):
            q=ri*2+m;anchor=anchors[pair['id'],q];valid=anchor['valid'];targets=((np.asarray(values,np.float32)-np.float32(anchor['original_mean']))/np.float32(anchor['original_std'])) if values is not None and valid else None;drop=float(min(targets[0],targets[-1])-min(targets[1:-1])) if targets is not None else None;records.append(dict(**pair,query=q,valid=valid,offered=values is not None,raw_scores=values,frozen_targets=targets.tolist() if targets is not None else None,worst_interior_drop_from_worse_endpoint=drop,strong_drop=drop>=.25 if drop is not None else None))
    summaries=[]
    for seed in [31,137,911]:
        for policy in ['parent','refined']:
            for secondary in [False,True]:
                rows=[r for r in records if r['seed']==seed and r['policy']==policy and (r['ratio_index']>=9)==secondary and r['valid']];offered=[r for r in rows if r['offered']];drops=[r['worst_interior_drop_from_worse_endpoint'] for r in offered];summaries.append(dict(seed=seed,policy=policy,secondary=secondary,valid_requests=len(rows),offered=len(offered),interior_below_worse_endpoint=sum(v>0 for v in drops),strong_drops=sum(r['strong_drop'] for r in offered),worst_drop=max(drops) if drops else None))
    write(a.output/'summary.json',dict(rows=summaries,pairs=len(pairs),distinct_scored_rectangles=len(raw),offbank_insertions=insertions,original_bank_bit_replays=replays,original_anchor_replays=replays*2,api_calls=0,test_inference=False,model_promoted=False))
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    progress(pid=None,stage='complete',completed=25);print(json.dumps(summaries))


if __name__=='__main__':main()
