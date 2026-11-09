#!/usr/bin/env python3
"""Nine bounded direct-window suitability controls; no shipping or old-gate claim."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time
import torch
from artifact_identity import sha
from fit_extents import retain
from fit_unsplash_ranks import alias
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from direct_crop import DirectCropHead,geometry,decode,loss
from checkpoint_snapshot import snapshot
from subject_geometry import pixels

EPOCHS=108
SEEDS=[31,137,911]
ARMS=['image','geometry-only','permuted-image']


def subset(context,indices):
    return dict(context,sizes=context['sizes'][indices],maximum=context['maximum'][indices],query=context['query'][indices])


@torch.no_grad()
def evaluate(model,visual,data,context,indices):
    model.eval();origins,scales,offer=model(visual[indices],context['query'][indices]);boxes,native=decode(origins,scales,subset(context,indices),integer=True)
    target=data['references'][indices];valid=data['reference_valid'][indices];known=valid.any(-1)
    error=((boxes[...,None,:]-target[:,:,None]).abs()*128).amax(-1).masked_fill(~valid[:,:,None],float('inf'))
    best=error.amin((-1,-2));first=error[:,:,0].amin(-1)
    faces=data['heads'][indices];face_valid=data['head_valid'][indices]
    b=native[:,:,:,None];f=faces[:,None,None]
    contained=(b[...,:2]<=f[...,:2]).all(-1)&(b[...,2:]>=f[...,2:]).all(-1)
    excluded=(b[...,2]<=f[...,0])|(f[...,2]<=b[...,0])|(b[...,3]<=f[...,1])|(f[...,3]<=b[...,1])
    cuts=(~contained&~excluded&face_valid[:,None,None]).any(-1)
    offer_valid=data['offer_valid'][indices];probability=offer.sigmoid();offered=probability>=.5
    values=dict(edge_error=best.masked_fill(~known,0),top1_edge_error=first.masked_fill(~known,0),
        best2_match=(best<=3).float(),top1_match=(first<=3).float(),top1_cut=cuts[:,:,0].float(),any_cut=cuts.any(-1).float(),
        offered=offered.float(),brier=(probability-data['offer'][indices]).square(),
        false_offer=(offered&~data['offer'][indices].bool()).float(),missed_offer=(~offered&data['offer'][indices].bool()).float())
    result={};all_rows=[]
    for scope in ['all','essential','secondary']:
        query=torch.ones(22,dtype=torch.bool,device=visual.device) if scope=='all' else torch.arange(22,device=visual.device)<18
        if scope=='secondary':query=~query
        masks=dict(reference=known&query[None],offer=offer_valid&query[None])
        summary={}
        for key,value in values.items():
            mask=masks['offer' if key in ['offered','brier','false_offer','missed_offer'] else 'reference']
            active=mask.any(-1);per_source=(value*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            summary[key]=float(per_source[active].mean()) if bool(active.any()) else None
        summary.update(reference_sources=int(masks['reference'].any(-1).sum()),reference_requests=int(masks['reference'].sum()),offer_requests=int(masks['offer'].sum()))
        result[scope]=summary
    cpu={key:value.cpu().tolist() for key,value in values.items()};v=known.cpu().tolist();ov=offer_valid.cpu().tolist()
    for j,i in enumerate(indices.tolist()):
        for q in range(22):
            if v[j][q] or ov[j][q]:all_rows.append(dict(id=data['sources'][i]['id'],family_id=data['sources'][i]['family_id'],query=q,essential=q<18,
                reference_available=v[j][q],offer_known=ov[j][q],**{k:cpu[k][j][q] for k in cpu}))
    return result,all_rows,dict(native=native.cpu(),offer=offer.cpu())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--region-cache',type=Path);p.add_argument('--region-variant',choices=['none','local-face','local-pose','local-face-pose','oracle-heads'])
    p.add_argument('--compact-states',action='store_true');a=p.parse_args()
    if (a.region_cache is None)!=(a.region_variant is None):raise ValueError('region cache and variant must be supplied together')
    if sha(a.cache)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('frozen direct-crop cache differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2 if a.region_cache else .35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);sources=cache['sources'];device='cuda'
    if len(sources)!=72 or len({s['family_id'] for s in sources})!=72:raise ValueError('fixed72 independent source families required')
    data={key:cache[key].to(device) for key in ['references','reference_valid','offer','offer_valid']};data['sources']=sources
    context=geometry([s['native_size'] for s in sources],cache['ratios'],device=device)
    maxheads=max(map(len,cache['head_rois']));heads=torch.zeros(72,maxheads,4);head_valid=torch.zeros(72,maxheads,dtype=torch.bool)
    for i,rois in enumerate(cache['head_rois']):
        w,h=sources[i]['native_size']
        for j,roi in enumerate(rois):heads[i,j]=torch.tensor(pixels(roi,w,h,outward=True));head_valid[i,j]=True
    data['heads']=heads.to(device);data['head_valid']=head_valid.to(device)
    normalized=(cache['visual']-cache['mean'])/cache['std']
    if a.region_cache:
        regions=torch.load(a.region_cache,map_location='cpu',weights_only=False)
        if regions['schema']!='source-region-context-v1' or regions['parent_direct_cache_sha256']!=sha(a.cache) or regions['sources']!=sources:raise ValueError('registered region source/parent contract differs')
        extra=regions['variants'][a.region_variant]
        if extra.shape!=(72,47) or not bool(torch.isfinite(extra).all()):raise ValueError('finite fixed-capacity source descriptors required')
        fit_rows=[i for i,s in enumerate(sources) if s['role']=='fit'];mean=extra[fit_rows].mean(0);std=extra[fit_rows].std(0,correction=0).clamp_min(.001)
        normalized=torch.cat((normalized,(extra-mean)/std),-1)
    normalized=normalized.to(device)
    fit=[i for i,s in enumerate(sources) if s['role']=='fit' and bool(cache['offer_valid'][i].any())]
    dev=torch.tensor([i for i,s in enumerate(sources) if s['role']=='dev'],device=device)
    if not fit or any(s['role'] not in ['fit','dev'] for s in sources):raise ValueError('nonempty FIT and fixed DEV required')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(cache_sha256=sha(a.cache),arms=ARMS,seeds=SEEDS,epochs=EPOCHS,fit_with_known_targets=len(fit),registered_fit=48,development=24,
        optimizer='AdamW lr1e-3 weight_decay1e-4 clip5 fused FP32 TF32off',source_batch=12,all_queries=True,source_balanced_losses=True,
        selection='minimum DEV family mean best-of-two maximum-edge error, then offer Brier, then earliest epoch',
        old_quality_gate_measured=False,model_promoted=False,purpose='editorial',paid_api_calls=0,test_inference=False,face_coverage_certified=False,
        source_sha256={name:sha(Path(name)) for name in ['training/direct_crop.py','training/checkpoint_snapshot.py','tools/composition/fit_direct_crop.py']})
    protocol.update(region_cache_sha256=sha(a.region_cache) if a.region_cache else None,region_variant=a.region_variant,
        visual_features=normalized.shape[1],compact_states=a.compact_states,cuda_allocator_fraction=.2 if a.region_cache else .35)
    retain(a.output/'protocol.json',protocol);status=dict(pid=os.getpid(),stage='initializing',completed=0,required=9)
    def update(**fields):
        status.update(fields,checked_unix=time.time())
        (a.output/'PROGRESS.json').write_text(json.dumps(status,indent=2)+'\n')
    def failure():
        if status['pid'] is not None:update(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure);reports=[]
    for arm in ARMS:
        visual=torch.zeros_like(normalized) if arm=='geometry-only' else normalized[cache['donors']] if arm=='permuted-image' else normalized
        for seed in SEEDS:
            torch.manual_seed(seed);model=DirectCropHead(visual=normalized.shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
            directory=a.output/(arm+'-seed'+str(seed));directory.mkdir();history=[];best=None
            for epoch in range(EPOCHS+1):
                update(stage='fitting',arm=arm,seed=seed,epoch=epoch,completed=len(reports))
                if epoch:
                    model.train();order=torch.tensor(fit)[torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch))].tolist()
                    for start in range(0,len(order),12):
                        indices=torch.tensor(order[start:start+12],device=device);optimizer.zero_grad(set_to_none=True)
                        origins,scales,offer=model(visual[indices],context['query'][indices]);prediction,_=decode(origins,scales,subset(context,indices))
                        value,active=loss(model,prediction,offer,data['references'][indices],data['reference_valid'][indices],data['offer'][indices],data['offer_valid'][indices],context['mode'])
                        if not bool(torch.isfinite(value)):raise ValueError('nonfinite direct-crop loss')
                        value.backward();model.clear_inactive_grads(**active);torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                metrics,rows,replay=evaluate(model,visual,data,context,dev)
                history.append(dict(epoch=epoch,metrics=metrics));key=(metrics['essential']['edge_error'],metrics['essential']['brier'],epoch)
                if best is None or key<best:
                    best=key;path=directory/f'weights-{epoch:03}.pt'
                    candidate=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed,epoch=epoch,metrics=metrics,
                        replay=replay,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    if a.compact_states:
                        best_state=snapshot(candidate);best_rows=rows
                    if not a.compact_states or epoch==0:
                        torch.save(candidate,path)
                        retain(directory/f'predictions-{epoch:03}.json',rows)
                        if not a.compact_states:
                            alias(directory/'best.pt',path);alias(directory/'best-predictions.json',directory/f'predictions-{epoch:03}.json')
                if epoch%36==0:print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_epoch=best[-1],metrics=metrics['essential'])),flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=EPOCHS,history=history,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),directory/'checkpoint-108.pt')
            if a.compact_states:
                torch.save(best_state,directory/'weights-selected.pt');retain(directory/'predictions-selected.json',best_rows)
                alias(directory/'best.pt',directory/'weights-selected.pt');alias(directory/'best-predictions.json',directory/'predictions-selected.json')
            state=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
            metrics,rows,replay=evaluate(model,visual,data,context,dev)
            if metrics!=state['metrics'] or rows!=json.loads((directory/'best-predictions.json').read_text()) or any(not torch.equal(replay[k],state['replay'][k]) for k in replay):raise ValueError('selected direct-crop native replay differs')
            retain(directory/'history.json',history);report=dict(arm=arm,seed=seed,epoch=state['epoch'],metrics=metrics,native_replay_exact=True,
                weights_sha256=sha(directory/'best.pt'),parameters=sum(p.numel() for p in model.parameters()))
            retain(directory/'summary.json',report);reports.append(report)
            allowance=(128 if a.compact_states else 512)*1024**2
            if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>allowance:raise ValueError('registered fit-evidence allowance exceeded')
    retain(a.output/'summary.json',reports);retain(a.output/'resources.json',dict(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    update(pid=None,stage='complete',completed=9)


if __name__=='__main__':main()
