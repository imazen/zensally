#!/usr/bin/env python3
"""Bounded equal-capacity continuous-pair and measured-attention suitability fits."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time
import torch
from artifact_identity import sha
from distill_attention_linear import write
from refinement_inputs import reference_match
from subject_policy import face_relation
from crop_corridors import audit
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from nested_crop_head import NestedCropHead,context,subset,decode,endpoint_loss
from checkpoint_snapshot import snapshot

ARMS=['geometry-only','image-only','aligned-distillation','permuted-distillation','measured-attention']
SEEDS=[31,137,911]
EPOCHS=64


def forward(model,data,ctx,indices,arm):
    visual=data['visual'][indices]
    if arm=='geometry-only':visual=torch.zeros_like(visual)
    return model(visual,ctx['query'][indices],data['attention'][indices] if arm=='measured-attention' else None)


def source_mean(value,mask):
    active=mask.any(-1);per_source=(value.masked_fill(~mask,0).sum(-1)/mask.sum(-1).clamp_min(1))
    return float(per_source[active].mean()) if active.any() else None


def validate(native,ctx):
    outer,inner=native.unbind(-2);size=native[...,2:]-native[...,:2];p,q=ctx['pairs'].unbind(-1)
    if not bool(torch.isfinite(native).all() and (native==native.round()).all() and (native[...,:2]>=0).all() and (native[...,2:]<=ctx['sizes'][:,None,None]).all() and (size>0).all() and (size[...,0]*q[None,:,None]==size[...,1]*p[None,:,None]).all() and (outer[...,:2]<=inner[...,:2]).all() and (inner[...,2:]<=outer[...,2:]).all() and (size[:,:,1,0]<size[:,:,0,0]).all() and (size[:,:,0]==ctx['maximum'][...,None]*ctx['pairs'][None]).all()):raise ValueError('native pair contract fails')


@torch.no_grad()
def evaluate(model,data,ctx,indices,arm):
    model.eval();logits,aux=forward(model,data,ctx,indices,arm);normalized,native=decode(logits,subset(ctx,indices),integer=True);validate(native,subset(ctx,indices));target=data['targets'][indices];valid=data['target_valid'][indices];pair_error=((normalized[:,:,None]-target).abs()*128).amax((-1,-2)).masked_fill(~valid,torch.inf).amin(-1);known=valid.any(-1);essential=torch.arange(11,device=normalized.device)<9;flat=normalized.flatten(1,2);reference=data['references'][indices];reference_valid=data['reference_valid'][indices];edge=((flat[:,:,None]-reference).abs()*128).amax(-1).masked_fill(~reference_valid,torch.inf).amin(-1);ref_known=reference_valid.any(-1);report=dict(primary_pair_edge_error=source_mean(pair_error,known&essential),all_pair_edge_error=source_mean(pair_error,known),paired_essential_requests=int((known&essential).sum()),essential_proximity_match=source_mean((edge<=3).float(),ref_known&(torch.arange(22,device=edge.device)<18)),auxiliary_mse=float((aux-data['attention'][indices]).square().mean()))
    return report,native.cpu()


def audited_predictions(native,cache,direct):
    rows=[]
    for i,s in enumerate(cache['sources']):
        sid=s['id'];w,h=s['native_size'];heads=cache['heads'][sid];head_known=s['annotation_valid'] and bool(heads)
        for ri,ratio in enumerate(cache['ratios']):
            outer,inner=native[i,ri].long().tolist();interval=audit(inner,outer,w,h,ratio,heads)['geometrically_admissible'] if head_known else None
            for m,box in enumerate([outer,inner]):
                q=ri*2+m;known=bool(cache['reference_valid'][i,q].any());rows.append(dict(id=sid,role=s['role'],family_id=s['family_id'],query=q,ratio=ratio,native=box,head_known=head_known,reference_known=known,reference_match=reference_match(box,cache['reference_options'].get((sid,q),[]),w,h),weak_cut=any(face_relation(box,h)=='cut' for h in heads),head_retained=any(face_relation(box,h)=='contained' for h in heads),interval_safe=interval,offer_known=bool(direct['offer_valid'][i,q]),teacher_contain=bool(direct['offer_valid'][i,q] and not direct['offer'][i,q]),offered=True))
    groups=[]
    for role in ['fit','dev']:
        for secondary in [False,True]:
            selected=[r for r in rows if r['role']==role and (r['query']>=18)==secondary];accepted=[r for r in selected if r['head_known'] and r['reference_known']];groups.append(dict(role=role,secondary=secondary,accepted_endpoints=len(accepted),matches=sum(r['reference_match'] for r in accepted),cuts=sum(r['weak_cut'] for r in accepted),head_retained=sum(r['head_retained'] for r in accepted),emitted=len(selected),emitted_teacher_contain=sum(r['teacher_contain'] for r in selected),accepted_families=len({r['family_id'] for r in accepted})))
    return groups,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','direct','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4' or sha(a.direct)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('registered source caches differ')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);direct=torch.load(a.direct,map_location='cpu',weights_only=False);sources=cache['sources'];fit=[i for i,s in enumerate(sources) if s['role']=='fit' and cache['target_valid'][i].any()];dev=[i for i,s in enumerate(sources) if s['role']=='dev'];all_indices=torch.arange(72,device='cuda')
    if len(sources)!=72 or len(fit)!=29 or len(dev)!=24 or len({s['family_id'] for s in sources})!=72 or any(s['role'] not in ['fit','dev'] for s in sources):raise ValueError('frozen source/family roles differ')
    data={k:cache[k].cuda() for k in ['visual','attention','permuted_attention','targets','target_valid','references','reference_valid']};ctx=context([s['native_size'] for s in sources],cache['ratios'],device='cuda');fit_tensor=torch.tensor(fit,device='cuda');dev_tensor=torch.tensor(dev,device='cuda');a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(cache_sha256=sha(a.cache),direct_sha256=sha(a.direct),arms=ARMS,seeds=SEEDS,epochs=EPOCHS,fit_sources=len(fit),dev_sources=len(dev),optimizer='AdamW',learning_rate=.001,weight_decay=.0001,clip=5,batch_sources=8,auxiliary_loss_weight=.1,selection='DEV essential source-balanced max-edge distance to closest existing exact same-story pair, then earliest epoch',api_calls=0,test_inference=False,new_teacher_inference=False,model_promoted=False));status=dict(pid=os.getpid(),stage='fitting',completed=0,required=15);reports=[]
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure)
    for seed in SEEDS:
        for arm in ARMS:
            path=a.output/f'{arm}-seed{seed}';path.mkdir();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=NestedCropHead().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);generator=torch.Generator().manual_seed(seed+1000);history=[];best=None;best_state=None;write(path/'architecture.json',dict(parameters=sum(p.numel() for p in model.parameters()),teacher_inputs_at_inference=arm=='measured-attention'))
            for epoch in range(EPOCHS+1):
                progress(seed=seed,arm=arm,epoch=epoch,completed=len(reports))
                if epoch:
                    model.train();order=torch.randperm(len(fit),generator=generator).tolist()
                    for start in range(0,len(order),8):
                        ix=fit_tensor[order[start:start+8]];optimizer.zero_grad(set_to_none=True);logits,predicted=forward(model,data,ctx,ix,arm);prediction,_=decode(logits,subset(ctx,ix));loss=endpoint_loss(prediction,data['targets'][ix],data['target_valid'][ix])
                        if arm not in ['geometry-only','image-only']:
                            expected=data['permuted_attention' if arm=='permuted-distillation' else 'attention'][ix];loss=loss+.1*(predicted-expected).square().mean()
                        if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                metric,replay=evaluate(model,data,ctx,dev_tensor,arm);fit_metric,_=evaluate(model,data,ctx,fit_tensor,arm);history.append(dict(epoch=epoch,fit=fit_metric,dev=metric));key=(metric['primary_pair_edge_error'],epoch)
                if best is None or key<best:
                    best=key;best_state=snapshot(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metrics=metric,fit_metrics=fit_metric,replay=replay,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),batch_rng=generator.get_state(),arm=arm,seed=seed))
                if epoch==0:torch.save(snapshot(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),batch_rng=generator.get_state())),path/'initial.pt')
            torch.save(snapshot(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=EPOCHS,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),batch_rng=generator.get_state())),path/'terminal.pt');torch.save(best_state,path/'selected.pt');write(path/'history.json',history);model.load_state_dict(best_state['model']);metric,replay=evaluate(model,data,ctx,dev_tensor,arm)
            if metric!=best_state['metrics'] or not torch.equal(replay,best_state['replay']):raise ValueError('selected native/metric replay differs')
            _,gpu_native=evaluate(model,data,ctx,all_indices,arm);groups,rows=audited_predictions(gpu_native,cache,direct);cpu=NestedCropHead();cpu.load_state_dict(best_state['model']);cpu.eval();cpu_data={k:v.cpu() for k,v in data.items()};cpu_ctx=context([s['native_size'] for s in sources],cache['ratios']);_,cpu_native=evaluate(cpu,cpu_data,cpu_ctx,torch.arange(72),arm);scale=cpu_ctx['sizes'][:,None,None].repeat(1,1,1,2);movement=(gpu_native-cpu_native).abs()*128/scale;numerical=dict(rectangles=72*11*2,changed_rectangles=int((gpu_native!=cpu_native).any(-1).sum()),maximum_edge_movement_grid=float(movement.max()),gate_violations=int((movement.amax(-1)>3).sum()))
            if numerical['gate_violations']:raise ValueError('unchanged three-grid numerical gate fails')
            write(path/'numerical.json',numerical);write(path/'selected-predictions.json',rows);report=dict(arm=arm,seed=seed,selected_epoch=best_state['epoch'],metrics=metric,fit_metrics=best_state['fit_metrics'],groups=groups,numerical=numerical,checkpoint_sha256=sha(path/'selected.pt'),parameters=sum(p.numel() for p in model.parameters()),teacher_inputs_at_inference=arm=='measured-attention');write(path/'summary.json',report);reports.append(report);write(a.output/'summary.json',reports);print(json.dumps(dict(arm=arm,seed=seed,epoch=report['selected_epoch'],dev=metric,accepted=next(g for g in groups if g['role']=='dev' and not g['secondary']),numerical=numerical)),flush=True)
            if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>128*1024*1024:raise ValueError('registered evidence budget exceeded')
    write(a.output/'resources.json',dict(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()));progress(pid=None,stage='complete',completed=15)


if __name__=='__main__':main()
