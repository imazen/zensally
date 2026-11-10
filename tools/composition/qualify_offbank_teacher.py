#!/usr/bin/env python3
"""Replay original FocalNet targets and test off-bank slot independence."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from cached_focalnet import CachedContext
from rank_targets import targets
from distill_attention_linear import write


def bits_equal(a,b):return np.array_equal(a.view(np.uint32),b.view(np.uint32))


def prepared(cache,index):
    bounds=cache['content'][index:index+1].numpy();x0,y0,x1,y1=np.rint(bounds[0]*256).astype(int);image=np.zeros((1,3,256,256),np.float32);mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None];image[0,:,y0:y1,x0:x1]=((cache['image'][index].numpy()-mean)/std)[:,y0:y1,x0:x1]
    return image,bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rank','packet','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.rank)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.packet/'SHA256SUMS.json')!='a49bbedcd4894ed7a7dd0c5a9f6b0f3a31647aba8f4e02d5675caa95e2e4d04e':raise ValueError('frozen teacher inputs differ')
    for name,digest in [('analysis.onnx','a5f322876a6d53b9221e1eca6d6a6b95e91b7b911e7003d660444492ca504768'),('query.onnx','8d14feeb5218baea8d885bba17e5b13b21e5c4dc6b6942a4d5de63adc51dc7ad')]:
        verify_member(a.packet,'runtime/split/'+name)
        if sha(a.packet/'runtime/split'/name)!=digest:raise ValueError('pinned original cached split differs')
    c=torch.load(a.rank,map_location='cpu',weights_only=False);indices=[i for i,s in enumerate(c['sources']) if s['role']=='dev']
    if len(indices)!=25 or any(s['role'] not in ['fit','dev'] for s in c['sources']):raise ValueError('original25 DEV only; TEST closed')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(rank_sha256=sha(a.rank),teacher_packet_sha256=sha(a.packet/'SHA256SUMS.json'),analysis_threads=4,query_threads=1,checks='original128-slot raw score bits and all frozen rank-target structures; two padding-slot placements of one deterministic off-bank probe',normalization='original eligible query mean/std frozen; original query validity retained',api_calls=0,test_inference=False,new_gradients=False,model_promoted=False));context=CachedContext(a.packet/'runtime/split',threads=4,query_threads=1);rows=[];anchors=[];status=dict(pid=os.getpid(),stage='teacher-replay',completed=0,required=25)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failed():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed)
    def require(ok,message,data):
        if not ok:write(a.output/'failure.json',dict(error=message,**data));raise ValueError(message)
    for number,index in enumerate(indices):
        source=c['sources'][index];image,bounds=prepared(c,index);token=context.analyze(image,bounds,(source['id'],source['rgb_sha256']));w,h=source['native_size'];progress(source=source['id'],completed=number)
        for ri,ratio in enumerate(c['ratios']):
            boxes=c['boxes'][index:index+1,ri].numpy().copy();reference=c['teacher_scores'][index,ri].numpy();actual=context.score(boxes,token)[0].copy();identity=dict(id=source['id'],ratio=ratio);require(bits_equal(actual,reference),'original cached teacher score bits differ',identity)
            for mode in range(2):
                q=ri*2+mode;selected=c['eligible'][index,q].nonzero().flatten().tolist();mask,valid,target,pair,pm=targets(actual,selected);require(np.array_equal(mask,c['eligible'][index,q].numpy()) and bool(valid)==bool(c['query_valid'][index,q]) and bits_equal(target,c['targets'][index,q].numpy()) and np.array_equal(pair,c['pairs'][index,q].numpy()) and np.array_equal(pm,c['pair_valid'][index,q].numpy()),'frozen target structures differ',dict(identity,query=q));anchors.append(dict(id=source['id'],query=q,valid=bool(valid),eligible=selected,original_mean=float(actual[selected].mean()) if selected else None,original_std=float(actual[selected].std()) if selected else None,original_best=float(actual[selected].max()) if selected else None))
            empty=np.flatnonzero((boxes[0]==0).all(-1));require(len(empty)>=2,'two original padding slots required',identity);p,r=map(int,ratio.split(':'));k=min(w//p,h//r)-1;probe=[0,0,k*p,k*r];require(k>0 and tuple(probe) not in {tuple(b) for b in c['native_boxes'][index][ri]},'deterministic off-bank probe invalid or already present',identity);probe_box=np.array(probe,np.float64)/[w,h,w,h];scores=[]
            for slot in empty[:2]:
                changed=boxes.copy();changed[0,slot]=probe_box;value=context.score(changed,token)[0].copy();keep=np.arange(128)!=slot;require(bits_equal(value[keep],reference[keep]),'off-bank insertion changes original candidate scores',dict(identity,slot=int(slot)));scores.append(value[slot])
            require(bits_equal(np.asarray(scores[:1]),np.asarray(scores[1:])),'off-bank probe depends on its padding slot',identity);rows.append(dict(identity,probe_native=probe,probe_slots=empty[:2].tolist(),probe_score=float(scores[0]),raw_scores_bit_exact=True,unmodified_slots_bit_exact=True,probe_slot_bits_exact=True,original_scores_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        progress(completed=number+1)
    write(a.output/'banks.json',rows);write(a.output/'normalization-anchors.json',anchors);write(a.output/'summary.json',dict(sources=25,original_banks=len(rows),original_raw_scores_checked=len(rows)*128,frozen_query_structures_checked=len(anchors),offbank_probe_placements=len(rows)*2,all_original_bits_exact=True,all_original_structures_exact=True,all_probe_slots_independent=True,scope='local cached teacher qualification only; no artistic or complete-face claim and no changed frozen-bank gate',api_calls=0,test_inference=False,model_promoted=False));progress(pid=None,stage='complete',completed=25);print(json.dumps(dict(sources=25,banks=len(rows),query_structures=len(anchors),probe_placements=len(rows)*2,all_bits_exact=True)))


if __name__=='__main__':main()
