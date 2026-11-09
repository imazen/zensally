#!/usr/bin/env python3
"""Nine fixed-encoder candidate-retention controls; TEST and face gates stay closed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from artifact_identity import sha
from fit_extents import retain,alias
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import family_weights

ARMS=['image','geometry','permuted']


def vector(data,indices,queries,arm):
    value=data['permuted' if arm=='permuted' else 'image'][indices,queries]
    if arm=='geometry':
        value=value.clone();value[...,:704]=0
    return value


def objective(raw,contained,fraction,valid):
    target=contained.float()
    loss=F.binary_cross_entropy_with_logits(raw[...,0],target,reduction='none')
    positive=valid&contained;negative=valid&~contained
    np,nn=positive.sum(-1),negative.sum(-1)
    binary=((loss*positive).sum(-1)/np.clamp_min(1)+(loss*negative).sum(-1)/nn.clamp_min(1))/((np>0).float()+(nn>0).float())
    mse=((raw[...,1].sigmoid()-fraction).square()*valid).sum(-1)/valid.sum(-1)
    return binary+.5*mse


@torch.no_grad()
def evaluate(head,data,indices,arm,sources):
    count=len(indices)
    source=torch.tensor(indices,device='cuda')[:,None].expand(-1,11).reshape(-1)
    query=torch.arange(11,device='cuda')[None].expand(count,-1).reshape(-1)
    raw=head(vector(data,source,query,arm)).reshape(count,11,128,2)
    if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite development retention logits')
    probability=raw[...,0].sigmoid();fraction=raw[...,1].sigmoid()
    valid=data['bank_valid'][indices];contained=data['contained'][indices]
    brier=((probability-contained.float()).square()*valid).sum(-1)/valid.sum(-1)
    mae=((fraction-data['fraction'][indices]).abs()*valid).sum(-1)/valid.sum(-1)
    weights=family_weights(sources,indices).cuda()
    metrics=dict(containment_brier=float((brier.mean(-1)*weights).sum()),
        fraction_mae=float((mae.mean(-1)*weights).sum()),sources=count,
        families=len({sources[i]['family_id'] for i in indices}))
    rows=[];decision_fields=[]
    for q in range(22):
        eligible=data['eligible'][indices,q]
        admissible=(probability[:,q//2]>=.5)&eligible
        offered=admissible.any(-1)
        selected=data['scores'][indices,q].masked_fill(~admissible,-torch.inf).argmax(-1)
        old=data['scores'][indices,q].masked_fill(~eligible,-torch.inf).argmax(-1)
        truth=contained[:,q//2].gather(1,selected[:,None]).squeeze(1)
        retained=data['fraction'][indices,q//2].gather(1,selected[:,None]).squeeze(1)
        opportunity=(contained[:,q//2]&eligible).any(-1)
        decision_fields.append(torch.stack((offered.float(),selected.float(),old.float(),truth.float(),retained,opportunity.float()),-1))
    decisions_cpu=torch.stack(decision_fields,1).cpu().tolist()
    for q in range(22):
        for j,i in enumerate(indices):
            offered,selected,old,truth,retained,opportunity=decisions_cpu[j][q]
            rows.append(dict(id=sources[i]['id'],family_id=sources[i]['family_id'],query=q,
                ratio=data['ratios'][q//2],mode='maximal' if q%2 else 'minimal',
                scope='essential' if q<18 else 'secondary',offered=bool(offered),
                selected=int(selected) if offered else None,old_selected=int(old),
                known_boxes_contained=bool(truth) if offered else None,
                retained_fraction=float(retained) if offered else None,
                bank_opportunity=bool(opportunity),face_coverage_certified=False))
    decisions={}
    for scope in ['all','essential','secondary']:
        chosen=[r for r in rows if scope=='all' or r['scope']==scope]
        decisions[scope]=dict(requests=len(chosen),offered=sum(r['offered'] for r in chosen),
            known_box_violations=sum(r['offered'] and not r['known_boxes_contained'] for r in chosen),
            recovered_opportunities=sum(r['offered'] and r['known_boxes_contained'] for r in chosen),
            available_opportunities=sum(r['bank_opportunity'] for r in chosen),
            unnecessary_abstentions=sum(not r['offered'] and r['bank_opportunity'] for r in chosen),
            off_bank_opportunity_offers=sum(r['offered'] and not r['bank_opportunity'] for r in chosen))
    metrics['decisions']=decisions
    return metrics,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=a.root/'features.pt'
    qualification=json.loads((a.root/'qualification.json').read_text())
    if sha(path)!=qualification['feature_sha256'] or qualification['max_original_score_error']!=0 or qualification['max_original_crop_map_edge_difference']>3:
        raise ValueError('qualified feature cache required')
    cache=torch.load(path,map_location='cpu',weights_only=False);sources=cache['sources']
    if cache['region_cache_sha256']!='101110157703882de1020b24df012d33f7a9adfd7b02cfb5c2f5588366c7918d' or cache['visual_dimensions']!=704:
        raise ValueError('registered known-box labels/layout differ')
    fit=[i for i,s in enumerate(sources) if s['role']=='fit'];dev=[i for i,s in enumerate(sources) if s['role']=='dev']
    if len(fit)!=72 or len(dev)!=15 or len(sources)!=87:raise ValueError('fixed fitting/development scope differs')
    if {sources[i]['family_id'] for i in fit}&{sources[i]['family_id'] for i in dev}:raise ValueError('source families cross roles')
    data={k:cache[k].cuda() for k in ['image','permuted','scores','eligible','bank_valid','contained','fraction']}
    data['ratios']=cache['ratios']
    if not bool(data['bank_valid'].any(-1).all()):raise ValueError('every known source/ratio needs valid candidate geometry')
    phase=torch.tensor([int.from_bytes(hashlib.sha256(('retention-phase-v1\0'+s['id']).encode()).digest()[:8],'big')%11 for s in sources])
    protocol=dict(feature_sha256=sha(path),region_cache_sha256=cache['region_cache_sha256'],arms=ARMS,
        seeds=[31,137,911],epochs=110,batch=16,lr=.001,weight_decay=.01,clip=5.,
        precision='FP32 TF32 off',encoder_frozen=True,geometry_zeroed_visual_dimensions=704,
        loss='equally weighted known positive/negative containment BCE +0.5 fraction sigmoid MSE',
        selection='minimum family-mean development containment Brier, then earlier epoch',
        crop_probability_threshold=.5,query_phases=phase.tolist(),
        labels='noisy known boxes on fitting sources only; not complete subjects or artistic labels',
        source_roles_changed=False,face_coverage_certified=False,new_test_inference=False,paid_model_api_calls=0)
    out=a.root/'fits';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('existing fit scientific protocol differs')
    else:retain(out/'protocol.json',protocol)
    weight=torch.zeros(87,device='cuda');weight[fit]=family_weights(sources,fit).cuda()*72
    reports=[]
    for arm in ARMS:
        for seed in protocol['seeds']:
            directory=out/(arm+'-seed'+str(seed));directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():
                report=json.loads((directory/'summary.json').read_text())
                if sha(directory/'best.pt')!=report['best_weights_sha256']:raise ValueError('completed selected head differs')
                reports.append(report);continue
            torch.manual_seed(seed)
            head=nn.Sequential(nn.Linear(762,96),nn.ReLU(),nn.Linear(96,2)).cuda()
            head[0].load_state_dict(cache['initial_head'])
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01,fused=True)
            history=[];best=float('inf');best_epoch=None;start=0;visits=torch.zeros(87,11,dtype=torch.int32)
            if (directory/'latest.pt').exists():
                state=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
                if state['protocol']!=protocol or state['arm']!=arm or state['seed']!=seed:raise ValueError('resume changes registered fit')
                head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
                history,best,best_epoch,start,visits=state['history'],state['best'],state['best_epoch'],state['epoch']+1,state['visits']
                torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            for epoch in range(start,111):
                if epoch:
                    order=torch.randperm(72,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                    for offset in range(0,72,16):
                        idx=torch.tensor([fit[i] for i in order[offset:offset+16]])
                        query=(phase[idx]+epoch-1)%11;visits[idx,query]+=1
                        i,q=idx.cuda(),query.cuda();optimizer.zero_grad(set_to_none=True)
                        raw=head(vector(data,i,q,arm))
                        loss=objective(raw,data['contained'][i,q],data['fraction'][i,q],data['bank_valid'][i,q])
                        value=(loss*weight[i]).sum()/16
                        if not bool(torch.isfinite(value)):raise ValueError('nonfinite retention objective')
                        value.backward();nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                metrics,predictions=evaluate(head,data,dev,arm,sources)
                history.append(dict(epoch=epoch,development=metrics));improved=metrics['containment_brier']<best
                if improved:best=metrics['containment_brier'];best_epoch=epoch
                if improved or epoch%25==0 or epoch==110:
                    checkpoint=directory/f'epoch-{epoch:03}.pt'
                    torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,arm=arm,seed=seed,
                        protocol=protocol,history=history,best=best,best_epoch=best_epoch,visits=visits,
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
                    alias(directory/'latest.pt',checkpoint)
                    if improved:alias(directory/'best.pt',checkpoint);retain(directory/f'predictions-{epoch:03}.json',predictions);alias(directory/'best-predictions.json',directory/f'predictions-{epoch:03}.json')
                (out/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),arm=arm,seed=seed,epoch=epoch,
                    completed_runs=len(reports),required_runs=9,required_epochs=110,checked_unix=time.time()))+'\n')
                if epoch%25==0:
                    print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_epoch=best_epoch,development=metrics)),flush=True)
                    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>2*1024**3:raise ValueError('two GiB fit-state budget exceeded; preserve files')
            if not bool((visits[fit]==10).all()) or bool(visits[dev].any()):raise ValueError('source-query visit or development-gradient contract differs')
            report=dict(arm=arm,seed=seed,best_epoch=best_epoch,best_dev=history[best_epoch]['development'],
                trained_parameters=sum(p.numel() for p in head.parameters()),best_weights_sha256=sha(directory/'best.pt'))
            retain(directory/'history.json',history);retain(directory/'summary.json',report);reports.append(report)
    retain(out/'summary.json',reports)
    (out/'progress.json').write_text(json.dumps(dict(pid=None,stage='complete',completed_runs=9,required_runs=9,checked_unix=time.time()))+'\n')
    print(json.dumps(dict(completed_runs=9,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())))


if __name__=='__main__':main()
