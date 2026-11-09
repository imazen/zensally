#!/usr/bin/env python3
"""Controlled weak acceptable-set fitting with old-rank replay and separate gates."""
import argparse
import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

import torch
from artifact_identity import sha
from fit_extents import retain
from fit_unsplash_ranks import alias,encode,evaluate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker,objective,ARMS,family_weights
from acceptable_crop_sets import acceptable_set_loss,reference_boxes

EPOCHS=44
SEEDS=[31,137,911]

@torch.no_grad()
def evaluate_sets(model,data,indices,donors,sources):
    model.eval();features=encode(model,data['image'],donors[indices]);rows=[]
    for q in range(22):
        boxes=reference_boxes(data,indices,q)
        raw=model.scores(features,boxes,data['source_aspect'][indices],data['aspect'][q].expand(len(indices)),data['mode'][q].expand(len(indices)),data['content'][indices])
        if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite weak-reference inference')
        selected=raw.masked_fill(~data['eligible'][indices,q],-torch.inf).argmax(-1)
        for j,i in enumerate(indices):
            if not bool(data['reference_valid'][i,q].any()):continue
            choice=int(selected[j]);rows.append(dict(id=sources[i]['id'],family_id=sources[i]['family_id'],query=q,
                covered=bool(data['positive'][i,q].any()),reference_match=bool(data['positive'][i,q,choice]),
                explicit_face_cut=bool(data['negative'][i,q,choice]),selected=choice,essential=q<18))
    if not rows:raise ValueError('development set has no valid weak crop reference')
    metrics={}
    for scope in ['all','essential','secondary']:
        groups={}
        for row in rows:
            if scope!='all' and row['essential']!=(scope=='essential'):continue
            groups.setdefault(row['family_id'],[]).append(row)
        if not groups:
            metrics[scope]=dict(families=0,requests=0);continue
        metrics[scope]=dict(families=len(groups),requests=sum(map(len,groups.values())),
            **{key:sum(sum(r[key] for r in group)/len(group) for group in groups.values())/len(groups) for key in ['covered','reference_match','explicit_face_cut']})
    return metrics,rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['old_cache','set_cache','weights','output']:p.add_argument(key,type=Path)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    p.add_argument('--new-loss-weight',type=float,choices=[0.,.25],default=.25)
    a=p.parse_args()
    if sha(a.old_cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('frozen old inputs differ')
    quality_sources=['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']
    for name in quality_sources:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):raise ValueError('frozen original quality code differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=torch.load(a.old_cache,map_location='cpu',weights_only=False);new=torch.load(a.set_cache,map_location='cpu',weights_only=False)
    if new['protocol']['reference_tolerance_grid']!=3 or old['ratios']!=new['ratios']:raise ValueError('fixed3grid and matching ratio contract required')
    sources=[dict(s,corpus='old',ratios=old['ratios']) for s in old['sources']]+[dict(s,corpus='new',ratios=new['ratios']) for s in new['sources']]
    nold=len(old['sources']);count=len(sources)
    if nold!=119 or Counter(s['role'] for s in new['sources'])!=Counter(fit=48,dev=24):raise ValueError('registered94/25 old and48/24 new split required')
    fitold=[i for i,s in enumerate(sources[:nold]) if s['role']=='fit'];devold=[i for i,s in enumerate(sources[:nold]) if s['role']=='dev']
    fitnew=[i+nold for i,s in enumerate(sources[nold:]) if s['role']=='fit' and bool((new['positive'][i].any(-1)&new['negative'][i].any(-1)).any())]
    devnew=[i+nold for i,s in enumerate(sources[nold:]) if s['role']=='dev']
    if not fitnew or {s['family_id'] for s in sources if s['role']=='fit'}&{s['family_id'] for s in sources if s['role']=='dev'}:raise ValueError('nonempty independent fitting families required')
    data={k:torch.cat([old[k],new[k]]).cuda() for k in ['image','content','boxes','eligible']}
    if 'query_boxes' in new:
        data['query_boxes']=torch.cat([old['boxes'].repeat_interleave(2,dim=1),new['query_boxes']]).cuda()
    for key in ['targets','pairs','pair_valid','query_valid']:
        data[key]=torch.cat([old[key],torch.zeros((len(new['sources']),)+old[key].shape[1:],dtype=old[key].dtype)]).cuda()
    for key in ['positive','negative','reference_valid']:
        data[key]=torch.cat([torch.zeros((nold,)+new[key].shape[1:],dtype=new[key].dtype),new[key]]).cuda()
    data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda')
    data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in old['ratios'] for _ in range(2)],device='cuda')
    data['mode']=torch.tensor([q%2 for q in range(22)],dtype=torch.float32,device='cuda')
    identity=torch.arange(count,device='cuda');permuted=list(range(count));groups={}
    contents=torch.cat([old['content'],new['content']])
    for i,s in enumerate(sources):groups.setdefault((s['corpus'],s['role'],tuple((contents[i]*256).round().int().tolist())),[]).append(i)
    for values in groups.values():
        ordered=sorted(values,key=lambda i:hashlib.sha256(('acceptable-set-permute|'+sources[i]['id']).encode()).hexdigest())
        for i,j in zip(ordered,ordered[1:]+ordered[:1]):permuted[i]=j
    phase=torch.tensor([int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%22 for s in sources])
    old_weight=torch.zeros(count,device='cuda');old_weight[fitold]=family_weights(sources,fitold).cuda()*len(fitold)
    initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model']
    base=LocalCropRanker().cuda();base.load_state_dict(initial);base.configure('trainable-image')
    baseline,_=evaluate(base,data,devold,identity,sources);baseline_new,_=evaluate_sets(base,data,devnew,identity,sources);del base
    protocol=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        quality_source_sha256={name:sha(Path(name)) for name in quality_sources},epochs=EPOCHS,seeds=SEEDS,arms=a.arms,new_loss_weight=a.new_loss_weight,old_cache_sha256=sha(a.old_cache),set_cache_sha256=sha(a.set_cache),initializer_sha256=sha(a.weights),
        old_fit=len(fitold),new_fit_with_contrastive_targets=len(fitnew),new_registered_fit=48,new_development=24,
        optimizer='AdamW lr1e-4 weight_decay1e-4 clip5 FP32 TF32off',source_batch='8old+8new, cyclic source-balanced cohorts',
        objective='family-balanced old rank loss +new_loss_weight times new multi-positive versus explicit-face-cut mass; unknown crops ignored by new term',
        selection='old essential regret<=fixed original initializer, then new supplied-face cuts, then reference match, then old regret and earliest epoch',
        baseline_old=baseline,baseline_new=baseline_new,permuted_sources=sum(i!=j for i,j in enumerate(permuted)),permutation_groups='same corpus/role/letterbox content',
        input_size=256,face_coverage_certified=False,artistic_gold=False,source_roles_changed=False,test_inference=False,
        new_bank=new['protocol'].get('bank','original-shared'),old_quality_replay_uses_original_boxes=True)
    a.output.mkdir(parents=True,exist_ok=False);retain(a.output/'protocol.json',protocol)
    status=dict(pid=os.getpid(),completed=0,required=len(a.arms)*len(SEEDS),stage='initializing')
    def update(**fields):status.update(fields,checked_unix=time.time());(a.output/'PROGRESS.json').write_text(json.dumps(status)+'\n')
    def finish_failure():
        if status['pid'] is not None:update(pid=None,stage='failed-artifacts-retained')
    atexit.register(finish_failure);reports=[]
    for arm in a.arms:
        donors=torch.tensor(permuted,device='cuda') if arm=='permuted-image' else identity
        for seed in SEEDS:
            torch.manual_seed(seed);directory=a.output/(arm+'-seed'+str(seed));directory.mkdir()
            model=LocalCropRanker().cuda();model.load_state_dict(initial);model.configure(arm)
            parameters=[v for v in model.parameters() if v.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4,fused=True)
            frozen=None
            if arm=='frozen-image':
                model.eval()
                with torch.no_grad():frozen=encode(model,data['image'],list(range(count)))
            history=[];best=None;bestpath=None
            for epoch in range(EPOCHS+1):
                update(stage='fitting',arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    model.train_arm();rng=torch.Generator().manual_seed(seed+epoch)
                    orders=[torch.tensor(pool)[torch.randperm(len(pool),generator=rng)].tolist() for pool in [fitold,fitnew]]
                    for start in range(0,max(map(len,orders)),8):
                        batches=[[order[(start+j)%len(order)] for j in range(8)] for order in orders]
                        optimizer.zero_grad(set_to_none=True);losses=[]
                        for cohort,raw_indices in enumerate(batches):
                            if cohort and a.new_loss_weight==0:continue
                            indices=torch.tensor(raw_indices,device='cuda');query=((phase[raw_indices]+epoch-1)%22).cuda()
                            if cohort:
                                valid=data['positive'][indices,query].any(-1)&data['negative'][indices,query].any(-1)
                                indices,query=indices[valid],query[valid]
                                if not len(indices):continue
                            features=tuple(f[indices] for f in frozen) if frozen is not None else model.feature_maps(data['image'][donors[indices]])
                            boxes=reference_boxes(data,indices,query) if cohort else data['boxes'][indices,query//2]
                            scores=model.scores(features,boxes,data['source_aspect'][indices],data['aspect'][query],data['mode'][query],data['content'][indices])
                            if cohort:
                                loss,_=acceptable_set_loss(scores,data['positive'][indices,query],data['negative'][indices,query]);losses.append(a.new_loss_weight*loss.mean())
                            else:losses.append((objective(scores,data['targets'][indices,query],data['pairs'][indices,query],data['pair_valid'][indices,query],data['eligible'][indices,query],data['query_valid'][indices,query])*old_weight[indices]).mean())
                        loss=sum(losses)
                        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite acceptable-set loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
                oldmetrics,oldpred=evaluate(model,data,devold,donors,sources);newmetrics,newpred=evaluate_sets(model,data,devnew,donors,sources)
                passes=oldmetrics['essential']['teacher_regret']<=baseline['essential']['teacher_regret']
                key=(not passes,newmetrics['essential']['explicit_face_cut'],-newmetrics['essential']['reference_match'],oldmetrics['essential']['teacher_regret'],epoch)
                history.append(dict(epoch=epoch,old=oldmetrics,new=newmetrics,old_nonregression=passes))
                if best is None or key<best:
                    best=key;bestpath=directory/f'weights-{epoch:03}.pt'
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed,epoch=epoch,protocol=protocol,
                        old=oldmetrics,new=newmetrics,old_nonregression=passes,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),bestpath)
                    retain(directory/f'predictions-{epoch:03}.json',dict(old=oldpred,new=newpred));alias(directory/'best.pt',bestpath);alias(directory/'best-predictions.json',directory/f'predictions-{epoch:03}.json')
                if epoch%22==0:print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_epoch=best[-1],old_nonregression=passes,new=newmetrics['essential'])),flush=True)
                if epoch==EPOCHS:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed,epoch=epoch,protocol=protocol,
                        history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),directory/'checkpoint-044.pt')
            state=torch.load(bestpath,map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
            om,op=evaluate(model,data,devold,donors,sources);nm,npred=evaluate_sets(model,data,devnew,donors,sources)
            if om!=state['old'] or nm!=state['new'] or dict(old=op,new=npred)!=json.loads((directory/'best-predictions.json').read_text()):raise ValueError('selected acceptable-set native replay differs')
            retain(directory/'history.json',history)
            report=dict(arm=arm,seed=seed,epoch=state['epoch'],old=om,new=nm,old_nonregression=state['old_nonregression'],replay_exact=True,weights_sha256=sha(bestpath))
            retain(directory/'summary.json',report);reports.append(report)
            if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>1024**3:raise ValueError('1GiB fit evidence cap exceeded')
    retain(a.output/'summary.json',reports);retain(a.output/'resources.json',dict(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    update(pid=None,stage='complete',completed=len(reports))

if __name__=='__main__':main()

