#!/usr/bin/env python3
"""Frozen risk-head proxy screen on broader people, without selecting on this panel."""
import argparse
import atexit
import os
import json
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import write
from broader_people_context import context
from region_risk_features import build,features,head_applicability
from candidate_cut_head import CutHead
from evaluate_person_risk import choose
from local_rank_distillation import family_weights
from subject_policy import face_relation


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','regions','risk','output']:p.add_argument(name,type=Path)
    p.add_argument('--attention',type=Path);p.add_argument('--attention-sha');a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='b027605dc44482458ba939eae54ab0035f95599701709b31c78f36cb9cfe7dd8' or sha(a.regions)!='72346b4a8ec6875eb047aa393b177c6aeb3c665b2eb49a0678e6310c1df4523a' or sha(a.risk/'SHA256SUMS.json')!='0fbcd218ec4012a72113ab809fa8c489455a9182a65c1f54caf559c9f8968080':raise ValueError('frozen broader inputs differ')
    for name in ['fits/protocol.json','fits/summary.json','routing/selected.json','head-gated-routing/selected.json']:verify_member(a.risk,name)
    if (a.attention is None)!=(a.attention_sha is None):raise ValueError('attention cache requires explicit hash')
    c=torch.load(a.cache,map_location='cpu',weights_only=False)
    if a.attention is not None:
        if sha(a.attention)!=a.attention_sha:raise ValueError('attention cache differs')
        auxiliary=torch.load(a.attention,map_location='cpu',weights_only=False)
        if auxiliary['sources']!=c['sources'] or auxiliary['schema']!='broader-candidate-attention-v1':raise ValueError('attention source/schema differs')
        c.update(extras=auxiliary['extras'],permuted_extras=auxiliary['permuted_extras'])
    regions=json.loads(a.regions.read_text());identities=json.loads((a.risk/'fits/protocol.json').read_text())['inputs'];ctx=context(c,regions,identities);proposal=dict(sources=c['sources'],groups=regions['groups'],donors=regions['donors']);real=build(ctx,proposal);permuted=build(ctx,proposal,permuted=True);fits=[r for r in json.loads((a.risk/'fits/summary.json').read_text())['fits'] if a.attention is not None or r['arm'] in ['geometry-regions','hidden-regions']]
    required=12 if a.attention is not None else 6
    if len(fits)!=required:raise ValueError('complete registered control subset required')
    a.output.mkdir(parents=True,exist_ok=False)
    terminal=[False]
    def failed():
        if not terminal[0]:write(a.output/'PROGRESS.json',dict(pid=None,stage='failed-artifacts-retained'))
    atexit.register(failed)
    write(a.output/'PROGRESS.json',dict(pid=os.getpid(),stage='evaluating',completed=0,required=required))
    write(a.output/'protocol.json',dict(inputs={str(a.cache):sha(a.cache),str(a.regions):sha(a.regions),str(a.risk/'SHA256SUMS.json'):sha(a.risk/'SHA256SUMS.json')},models_evaluated=required,total_risk_models=12,missing=None if a.attention is not None else 'six attention-input controls require actual broader gaze/scanpath features; no zero fabrication',attention_cache_sha256=a.attention_sha,sources=16,roles='DEV only; prior broad-rank/source-preview exposure',selection_on_this_panel=False,api_calls=0,test_inference=False,model_promoted=False));records=[];summaries=[];numerical=[]
    def retain(name,applicability,choice,present):
        flat=choice.reshape(16,22);offered=present.repeat_interleave(2,1);native=ctx['native'].reshape(16,22,128,4);source_heads=[sorted(set(map(tuple,[*g['faces'],*g['pose']]))) for g in regions['groups']]
        for i,s in enumerate(c['sources']):
            for q in range(22):
                k=int(flat[i,q]);box=native[i,q,k].tolist() if offered[i,q] else None;relations=[face_relation(box,list(h)) for h in source_heads[i]] if box is not None else [];best=c['targets'][i,q].masked_fill(~c['eligible'][i,q],-torch.inf).max();regret=float(best-c['targets'][i,q,k]) if offered[i,q] and c['query_valid'][i,q] else None;records.append(dict(model=name,applicability=applicability,id=s['id'],family_id=s['family_id'],group_cell=s['group_cell'],query=q,ratio=c['ratios'][q//2],offered=bool(offered[i,q]),selected=k if box is not None else None,native=box,query_valid=bool(c['query_valid'][i,q]),teacher_regret=regret,proposed_heads_known=bool(source_heads[i]),supplied_head_cut='cut' in relations,supplied_head_retained='contained' in relations))
        these=[r for r in records if r['model']==name and r['applicability']==applicability]
        for secondary in [False,True]:
            subset=[r for r in these if (r['query']>=18)==secondary];groups={}
            for row in subset:
                if row['teacher_regret'] is not None:groups.setdefault(row['family_id'],[]).append(row['teacher_regret'])
            summaries.append(dict(model=name,applicability=applicability,secondary=secondary,requests=len(subset),offered=sum(r['offered'] for r in subset),valid_proxy_requests=sum(r['teacher_regret'] is not None for r in subset),proxy_regret=sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None,source_families=len(groups),supplied_head_known_requests=sum(r['proposed_heads_known'] for r in subset),supplied_head_cut_requests=sum(r['supplied_head_cut'] for r in subset),supplied_head_retained_requests=sum(r['supplied_head_retained'] for r in subset),face_coverage_certified=False))
    empty=torch.full((16,11,2,128),-1.);baseline=choose(ctx,empty,'all',0.);retain('unfiltered','all',*baseline)
    for entry in fits:
        name=f"{entry['arm']}-seed{entry['seed']}";member='fits/'+name+'/selected.pt';verify_member(a.risk,member);path=a.risk/member
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('risk checkpoint differs')
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);model.eval();x=features(c,real,permuted,entry['arm']);reference=model(x).reshape(16,11,2,128);compact=model.compact(x).reshape_as(reference);gpu=model.cuda()(x.cuda()).cpu().reshape_as(reference);model.cpu()
        for gate,directory in [('person','routing'),('head','head-gated-routing')]:
            active=dict(ctx)
            if gate=='head':head_applicability({'panel':active},{'panel':proposal})
            selected=next(r for r in json.loads((a.risk/directory/'selected.json').read_text()) if r['model']==name);strength=selected['strength'];ref=choose(active,reference,'person',strength)
            if not ref[1].all():raise ValueError('broader soft policy unexpectedly withdrew offers')
            retain(name,gate,*ref)
            for implementation,value in [('compact',compact),('cuda',gpu)]:
                choice,present=choose(active,value,'person',strength);before,offered=ref;changed=(choice!=before).any(-1)&present&offered;presence=int((present!=offered).sum());edge=0.;violations=presence
                for i,ri in changed.nonzero().tolist():
                    movement=max(float(((ctx['native'][i,ri,m,choice[i,ri,m]]-ctx['native'][i,ri,m,before[i,ri,m]]).abs().double()/ctx['scale'][i,0,0]*128).max()) for m in range(2));edge=max(edge,movement);violations+=movement>3
                numerical.append(dict(model=name,applicability=gate,implementation=implementation,pair_requests=176,endpoint_changes=int((choice!=before).sum()),presence_changes=presence,max_edge_movement_grid=edge,gate_violations=violations))
        write(a.output/'PROGRESS.json',dict(pid=os.getpid(),stage='evaluating',completed=len(numerical)//4,required=required));print(json.dumps([r for r in summaries if r['model']==name and not r['secondary']]),flush=True)
    if any(r['gate_violations'] for r in numerical):raise ValueError('unchanged numeric crop gate fails')
    write(a.output/'predictions.json',records);write(a.output/'summary.json',dict(models_evaluated=required,total_risk_models=12,metrics=summaries,numerical=numerical,api_calls=0,test_inference=False,model_promoted=False));write(a.output/'PROGRESS.json',dict(pid=None,stage='complete',completed=required,required=required));terminal[0]=True;print(json.dumps(dict(baseline=summaries[:2],numerical_endpoint_changes=sum(r['endpoint_changes'] for r in numerical))))


if __name__=='__main__':main()
