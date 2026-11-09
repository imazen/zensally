#!/usr/bin/env python3
"""Fit old/new photographic rankings with corpus-separated development controls."""
import argparse
import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from artifact_identity import sha
from fit_extents import retain
from fit_unsplash_ranks import alias,encode,evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import ARMS,LocalCropRanker,family_weights,objective

EPOCHS=88
SEEDS=[31,137,911]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old_cache','new_cache','weights','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.old_cache,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),
        (a.new_cache,'420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d'),
        (a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df')]:
        if sha(path)!=digest:raise ValueError('registered broader-rank input differs')
    quality_sources=['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py',
        'training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']
    for name in quality_sources:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):
            raise ValueError('original quality source changed')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=torch.load(a.old_cache,map_location='cpu',weights_only=False);new=torch.load(a.new_cache,map_location='cpu',weights_only=False)
    if old['ratios']!=new['ratios']:raise ValueError('registered ratio sets differ')
    sources=[dict(s,corpus='old',ratios=old['ratios']) for s in old['sources']]+[dict(s,corpus='new',ratios=new['ratios']) for s in new['sources']]
    counts=Counter((s['corpus'],s['role']) for s in sources)
    if counts!=Counter({('old','fit'):94,('old','dev'):25,('new','fit'):275,('new','dev'):91}) or len({s['id'] for s in sources})!=485:
        raise ValueError('frozen combined369-fit/116-development sources required')
    fit=[i for i,s in enumerate(sources) if s['role']=='fit']
    dev={corpus:[i for i,s in enumerate(sources) if s['role']=='dev' and s['corpus']==corpus] for corpus in ['old','new']}
    if {s['family_id'] for s in sources if s['role']=='fit'} & {s['family_id'] for s in sources if s['role']=='dev'}:
        raise ValueError('source component crosses fit/development')
    data={key:torch.cat([old[key],new[key]]).cuda() for key in ['image','content','boxes','targets','pairs','pair_valid','eligible','query_valid']}
    data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda')
    data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in old['ratios'] for _ in range(2)],device='cuda')
    data['mode']=torch.tensor([i%2 for i in range(22)],dtype=torch.float32,device='cuda')
    phase=torch.tensor([int.from_bytes(hashlib.sha256(('query-phase-v1\0'+s['id']).encode()).digest()[:8],'big')%22 for s in sources])
    donors=list(range(485));groups={};content=torch.cat([old['content'],new['content']])
    for i,s in enumerate(sources):groups.setdefault((s['corpus'],s['role'],tuple((content[i]*256).round().int().tolist())),[]).append(i)
    rng=torch.Generator().manual_seed(31)
    for indices in groups.values():
        order=[indices[j] for j in torch.randperm(len(indices),generator=rng).tolist()]
        for i,j in zip(order,order[1:]+order[:1]):donors[i]=j
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        old_cache_sha256=sha(a.old_cache),new_cache_sha256=sha(a.new_cache),initializer_sha256=sha(a.weights),
        quality_source_sha256={name:sha(Path(name)) for name in quality_sources},arms=ARMS,seeds=SEEDS,epochs=EPOCHS,
        source_batch=16,lr=.001,weight_decay=.01,clip=5.,precision='FP32 TF32 off',
        objective='original family-balanced strong-pair logistic+0.25 masked score MSE',
        selection='equal-corpus essential family-mean regret, then higher pair agreement, then earliest epoch',
        query_phases=phase.tolist(),donors=donors,
        permutation_coverage={corpus+'_'+role:dict(sources=sum(s['corpus']==corpus and s['role']==role for s in sources),
            moved=sum(i!=donors[i] for i,s in enumerate(sources) if s['corpus']==corpus and s['role']==role),
            cross_family=sum(s['family_id']!=sources[donors[i]]['family_id'] for i,s in enumerate(sources) if s['corpus']==corpus and s['role']==role)) for corpus in ['old','new'] for role in ['fit','dev']},
        new_test_inference=False,human_artistic_quality=False,face_coverage_certified=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=False);retain(a.output/'protocol.json',protocol)
    status=dict(pid=os.getpid(),stage='initializing',completed_fits=0,required_fits=12)
    def update(**fields):status.update(fields,checked_unix=time.time());(a.output/'progress.json').write_text(json.dumps(status)+'\n')
    def failed():
        if status['pid'] is not None:update(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed);update()
    known=torch.cat([old['query_valid'],new['query_valid']]);weight=torch.zeros(485,device='cuda')
    if not bool(known[fit].any(-1).all()):raise ValueError('every fitting source needs a known ranking condition')
    weight[fit]=family_weights(sources,fit).cuda()*len(fit)*22/data['query_valid'][fit].sum(-1)
    initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];reports=[]
    for arm in ARMS:
        visual_donors=torch.tensor(donors if arm=='permuted-image' else list(range(485)),device='cuda')
        for seed in SEEDS:
            directory=a.output/(arm+'-seed'+str(seed));directory.mkdir();torch.manual_seed(seed)
            model=LocalCropRanker().cuda();model.load_state_dict(initial);model.configure(arm)
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01,fused=True)
            frozen=None
            if arm=='frozen-image':model.eval();frozen=encode(model,data['image'],list(range(485)))
            visits=torch.zeros(485,22,dtype=torch.int32);history=[];best=None;best_path=None
            for epoch in range(EPOCHS+1):
                update(stage='fitting',arm=arm,seed=seed,epoch=epoch,completed_fits=len(reports))
                if epoch:
                    model.train_arm();order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                    for offset in range(0,len(fit),16):
                        raw_indices=torch.tensor([fit[j] for j in order[offset:offset+16]]);query=(phase[raw_indices]+epoch-1)%22
                        visits[raw_indices,query]+=1;keep=known[raw_indices,query];indices=raw_indices[keep].cuda();q=query[keep].cuda()
                        if not len(indices):continue
                        optimizer.zero_grad(set_to_none=True)
                        features=tuple(f[indices] for f in frozen) if frozen is not None else model.feature_maps(data['image'][visual_donors[indices]])
                        raw=model.scores(features,data['boxes'][indices,q//2],data['source_aspect'][indices],data['aspect'][q],data['mode'][q],data['content'][indices])
                        loss=(objective(raw,data['targets'][indices,q],data['pairs'][indices,q],data['pair_valid'][indices,q],data['eligible'][indices,q],data['query_valid'][indices,q])*weight[indices]).sum()/16
                        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite broader-rank objective')
                        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
                metrics={};predictions=[]
                for corpus in ['old','new']:
                    metrics[corpus],values=evaluate(model,data,dev[corpus],visual_donors,sources)
                    predictions.extend([dict(row,corpus=corpus) for row in values])
                history.append(dict(epoch=epoch,development=metrics))
                key=(sum(metrics[c]['essential']['teacher_regret'] for c in ['old','new'])/2,
                    -sum(metrics[c]['essential']['strong_pair_accuracy'] for c in ['old','new'])/2,epoch)
                if best is None or key<best:
                    best=key;best_path=directory/f'weights-{epoch:03}.pt'
                    torch.save(dict(model=model.state_dict(),arm=arm,seed=seed,epoch=epoch,protocol=protocol,metrics=metrics),best_path)
                    retain(directory/f'predictions-{epoch:03}.json',predictions);alias(directory/'best.pt',best_path)
                    alias(directory/'best-predictions.json',directory/f'predictions-{epoch:03}.json')
                if epoch in [0,44,88]:
                    path=directory/f'checkpoint-{epoch:03}.pt'
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed,epoch=epoch,
                        protocol=protocol,history=history,visits=visits,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
                    alias(directory/'latest.pt',path)
                if epoch%22==0:
                    print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_epoch=best[-1],development={c:metrics[c]['essential'] for c in ['old','new']})),flush=True)
                    if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>2*1024**3:raise ValueError('2GiB broader fit artifact budget reached')
            if not bool((visits[fit]==4).all()) or bool(visits[dev['old']+dev['new']].any()):raise ValueError('source/query visit contract differs')
            state=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);replay={};replay_predictions=[]
            for corpus in ['old','new']:
                replay[corpus],values=evaluate(model,data,dev[corpus],visual_donors,sources);replay_predictions.extend([dict(row,corpus=corpus) for row in values])
            if replay!=state['metrics'] or replay_predictions!=json.loads((directory/'best-predictions.json').read_text()):raise ValueError('selected native replay differs')
            retain(directory/'history.json',history)
            report=dict(arm=arm,seed=seed,best_epoch=best[-1],best_dev=replay,best_weights_sha256=sha(best_path),
                selected_replay_exact=True,source_query_visits_exact=True,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters))
            retain(directory/'summary.json',report);reports.append(report)
    retain(a.output/'summary.json',reports);retain(a.output/'resources.json',dict(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    update(pid=None,stage='complete',completed_fits=12)
    print(json.dumps(dict(completed_fits=12,**json.loads((a.output/'resources.json').read_text()))))


if __name__=='__main__':main()
