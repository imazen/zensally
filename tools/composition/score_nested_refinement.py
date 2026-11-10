#!/usr/bin/env python3
"""Additional off-bank crop diagnostic using unchanged original teacher anchors."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from artifact_identity import sha
from cached_focalnet import CachedContext
from qualify_offbank_teacher import prepared,bits_equal
from local_rank_distillation import family_weights
from distill_attention_linear import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rank','packet','qualification','geometry','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.qualification/'protocol.json').read_text());qualification=json.loads((a.qualification/'summary.json').read_text())
    if sha(a.rank)!=protocol['rank_sha256'] or sha(a.packet/'SHA256SUMS.json')!=protocol['teacher_packet_sha256'] or not qualification['all_original_bits_exact']:raise ValueError('qualified teacher identity differs')
    for name,key in [('analysis.onnx','analysis'),('query.onnx','query')]:
        expected={'analysis':'a5f322876a6d53b9221e1eca6d6a6b95e91b7b911e7003d660444492ca504768','query':'8d14feeb5218baea8d885bba17e5b13b21e5c4dc6b6942a4d5de63adc51dc7ad'}[key]
        if sha(a.packet/'runtime/split'/name)!=expected:raise ValueError('qualified split changed')
    c=torch.load(a.rank,map_location='cpu',weights_only=False);indices=[i for i,s in enumerate(c['sources']) if s['role']=='dev'];sources=[c['sources'][i] for i in indices]
    if len(indices)!=25:raise ValueError('original DEV source count differs')
    rows=[json.loads(line) for line in (a.geometry/'predictions.jsonl').read_text().splitlines() if line];rows=[r for r in rows if r['cohort']=='old'];parents=[json.loads(line) for line in (a.geometry/'parents.jsonl').read_text().splitlines() if line];parents={(r['seed'],r['id'],r['ratio_index']):r for r in parents if r['cohort']=='old'}
    anchors={(r['id'],r['query']):r for r in json.loads((a.qualification/'normalization-anchors.json').read_text())};requested={};raw={};bank_replays=0;insertions=0;target_replays=0
    for r in rows:
        for m,key in enumerate(['outer','inner']):
            b=r[key]
            if b is not None:requested.setdefault((r['id'],r['ratio_index']),set()).add(tuple(b))
    for r in parents.values():
        for key in ['outer','inner']:requested.setdefault((r['id'],r['ratio_index']),set()).add(tuple(r[key]))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):sha(path) for path in [a.rank,a.packet/'SHA256SUMS.json',a.qualification/'protocol.json',a.qualification/'normalization-anchors.json',a.geometry/'predictions.jsonl',a.geometry/'parents.jsonl']},normalization='original eligible mean/std/best, validity and family weights; original candidates remain fixed',scope='additional off-bank teacher diagnostic; does not replace established frozen-bank gate',api_calls=0,test_inference=False,new_gradients=False,model_promoted=False));status=dict(pid=os.getpid(),stage='offbank-scoring',completed=0,required=25)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);context=CachedContext(a.packet/'runtime/split',threads=4,query_threads=1)
    for number,i in enumerate(indices):
        s=c['sources'][i];sid=s['id'];w,h=s['native_size'];image,bounds=prepared(c,i);token=context.analyze(image,bounds,(sid,s['rgb_sha256']));progress(source=sid,completed=number)
        for ri,ratio in enumerate(c['ratios']):
            boxes=c['boxes'][i:i+1,ri].numpy().copy();reference=c['teacher_scores'][i,ri].numpy();actual=context.score(boxes,token)[0].copy()
            if not bits_equal(actual,reference):raise ValueError('original raw teacher scores changed')
            bank_replays+=1;native=c['native_boxes'][i][ri];existing={tuple(b):float(reference[k]) for k,b in enumerate(native)}
            for m in range(2):
                q=ri*2+m;anchor=anchors[sid,q];eligible=c['eligible'][i,q].numpy();valid=bool(c['query_valid'][i,q])
                if anchor['valid']!=valid:raise ValueError('original validity changed')
                if valid:
                    replay=(reference[eligible]-np.float32(anchor['original_mean']))/np.float32(anchor['original_std'])
                    if not bits_equal(replay,c['targets'][i,q].numpy()[eligible]):raise ValueError('frozen anchor z-score replay differs')
                target_replays+=1
            for box in sorted(requested.get((sid,ri),set())):
                if box in existing:value=existing[box]
                else:
                    slot=int(np.flatnonzero((boxes[0]==0).all(-1))[0]);changed=boxes.copy();changed[0,slot]=np.asarray(box,np.float64)/[w,h,w,h];value_array=context.score(changed,token)[0].copy();keep=np.arange(128)!=slot
                    if not bits_equal(value_array[keep],reference[keep]):raise ValueError('actual refinement insertion changes original score bits')
                    value=float(value_array[slot]);insertions+=1
                raw[sid,ri,box]=value
        progress(completed=number+1)
    lookup={s['id']:i for i,s in enumerate(c['sources'])};predictions=[];summaries=[]
    def score(sid,ri,m,box):
        anchor=anchors[sid,ri*2+m]
        if box is None or not anchor['valid']:return None
        value=np.float32(raw[sid,ri,tuple(box)]);target=(value-np.float32(anchor['original_mean']))/np.float32(anchor['original_std']);i=lookup[sid];eligible=c['eligible'][i,ri*2+m];best=c['targets'][i,ri*2+m][eligible].max().item()
        return dict(raw=float(value),target=float(target),regret=float(np.float32(best)-target))
    for seed in [31,137,911]:
        for policy in sorted({r['policy'] for r in rows}):
            selected=[r for r in rows if r['seed']==seed and r['policy']==policy];regrets=torch.zeros(25,22);before=torch.zeros_like(regrets);available=torch.zeros(25,22,dtype=torch.bool);valid=c['query_valid'][indices];source_index={s['id']:i for i,s in enumerate(sources)}
            for r in selected:
                sid=r['id'];ri=r['ratio_index'];parent=parents[seed,sid,ri];i=source_index[sid]
                for m,key in enumerate(['outer','inner']):
                    q=ri*2+m;actual=score(sid,ri,m,r[key]);baseline=score(sid,ri,m,parent[key]);predictions.append(dict(seed=seed,policy=policy,id=sid,ratio=r['ratio'],query=q,offered=r[key] is not None,valid=bool(valid[i,q]),before=baseline,after=actual))
                    if baseline is not None:before[i,q]=baseline['regret']
                    if actual is not None:available[i,q]=True;regrets[i,q]=actual['regret']
            for secondary in [False,True]:
                band=slice(18,None) if secondary else slice(None,18);mask=valid[:,band];offered=available[:,band]&mask;has=offered.any(-1);which=has.nonzero().flatten().tolist();weights=family_weights(sources,which)
                value=float((((regrets[:,band][has]*offered[has]).sum(-1)/offered[has].sum(-1))*weights).sum()) if which else None
                original=float((((before[:,band][has]*offered[has]).sum(-1)/offered[has].sum(-1))*weights).sum()) if which else None
                summaries.append(dict(seed=seed,policy=policy,secondary=secondary,valid_requests=int(mask.sum()),offered=int(offered.sum()),conditional_frozen_anchor_regret=value,parent_same_support_regret=original,full_frozen_anchor_regret=value if offered[mask].all() else None,scope='off-bank additional diagnostic, not original fixed-bank gate'))
    write(a.output/'summary.json',dict(rows=summaries,original_bank_bit_replays=bank_replays,original_target_anchor_replays=target_replays,actual_offbank_insertions=insertions,distinct_scored_rectangles=len(raw),api_calls=0,test_inference=False,model_promoted=False))
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in predictions:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    progress(pid=None,stage='complete',completed=25);print(json.dumps(dict(primary=[r for r in summaries if r['policy']=='faces-pose/pad25'],bank_replays=bank_replays,target_replays=target_replays,offbank_insertions=insertions)))


if __name__=='__main__':main()
