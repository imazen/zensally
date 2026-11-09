#!/usr/bin/env python3
"""Audit nominated-head coverage beyond accepted composition references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from pair_relationship_head import PairCorrection
from fit_pair_relationship import choices
from subject_policy import face_relation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import pixels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','direct','endpoint','pairs','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    expected=[(a.cache,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.direct,'2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472'),(a.endpoint,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'),(a.pairs/'SHA256SUMS.json','698972129157fa6ffce1c748f94bacac12a42b703ddea7c300b5445743b93020')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('frozen coverage inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);d=torch.load(a.direct,map_location='cpu',weights_only=False)
    if c['sources']!=d['sources'] or any(s['role'] not in ['fit','dev'] for s in c['sources']):raise ValueError('fixed FIT/DEV identity required')
    state=torch.load(a.endpoint,map_location='cpu',weights_only=False)['model'];scalar=Student({k:state['base.'+k] for k in ['weight','bias']},state['teacher'],state['mean'],state['std'],nonlinear=True);scalar.load_state_dict(state);scalar.eval()
    scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().long();eligible=c['eligible'].reshape(72,11,2,128);o=native[:,::2,:,None,:];inner=native[:,1::2,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);nested=(o[...,:2]<=inner[...,:2]).all(-1)&(o[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&eligible[...,0,:,None]&eligible[...,1,None,:]
    with torch.no_grad():
        scores=scalar(c['hidden'],None);flat=(scores[:,::2,:,None].double()+scores[:,1::2,None,:].double()).masked_fill(~nested,-torch.inf).flatten(-2).argmax(-1);selected={'raw':scores.masked_fill(~c['eligible'],-torch.inf).argmax(-1),'joint':torch.stack([flat//128,flat%128],-1).reshape(72,22)}
        for name in ['additive-linear-seed31','pair-real-seed137']:
            arm=name.rsplit('-seed',1)[0];model=PairCorrection(arm,state['mean'],state['std']);path=a.pairs/name/'selected.pt';model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);selected[name]=choices(model,c['hidden'].reshape(72,11,2,128,96),scores.reshape(72,11,2,128),eligible,nested).reshape(72,22)
    queries=[];predictions=[];geometry_safe=torch.zeros_like(c['eligible']);heads_retained=torch.zeros_like(c['eligible']);negative_replay_changes=0;positive_cut_conflicts=0;positive_no_head=0;sources=[]
    for i,source in enumerate(c['sources']):
        w,h=source['native_size'];heads=[pixels(r,w,h,outward=True) for r in d['head_rois'][i]];head_known=source['annotation_valid'] and bool(heads)
        sources.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],annotation_valid=source['annotation_valid'],head_count=len(heads),head_natives=heads,source=source['source']))
        for q in range(22):
            e=c['eligible'][i,q];boxes=native[i,q].tolist();relations=[([face_relation(box,head) for head in heads] if active else []) for box,active in zip(boxes,e.tolist())];cut=torch.tensor(['cut' in r for r in relations])&e;retained=torch.tensor(['contained' in r for r in relations])&e
            negative_replay_changes+=int((cut!=c['negative'][i,q]).sum());positive_cut_conflicts+=int((cut&c['positive'][i,q]).sum());positive_no_head+=int((c['positive'][i,q]&~retained).sum());geometry_safe[i,q]=e&~cut;heads_retained[i,q]=retained
            offer_known=bool(d['offer_valid'][i,q]);recommendation=('crop' if d['offer'][i,q] else 'contain') if offer_known else 'unknown';reference=bool(c['reference_valid'][i,q].any())
            queries.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],query=q,essential=q<18,head_known=head_known,reference_known=reference,recommendation=recommendation,eligible=int(e.sum()),cut_candidates=int(cut.sum()),weak_head_safe_candidates=int((e&~cut).sum()),safe_retaining_head_candidates=int((e&~cut&retained).sum()),positive_candidates=int(c['positive'][i,q].sum())))
            for model,choice in selected.items():
                k=int(choice[i,q]);predictions.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],query=q,essential=q<18,head_known=head_known,reference_known=reference,recommendation=recommendation,model=model,selected=k,weak_head_cut=bool(cut[k]),weak_heads_retained=relations[k].count('contained'),weak_heads_excluded=relations[k].count('excluded'),weak_heads_cut=relations[k].count('cut'),reference_match=bool(c['positive'][i,q,k])))
    if negative_replay_changes or positive_cut_conflicts:raise ValueError('frozen weak label geometry disagrees')
    pair_safe=nested&geometry_safe[:,::2,:,None]&geometry_safe[:,1::2,None,:];pair_subject=pair_safe&heads_retained[:,1::2,None,:];pair_rows=[]
    for i,source in enumerate(c['sources']):
        for ri,ratio in enumerate(c['ratios']):pair_rows.append(dict(id=source['id'],role=source['role'],ratio=ratio,essential=ri<9,head_known=source['annotation_valid'] and bool(d['head_rois'][i]),weak_safe_pair_available=bool(pair_safe[i,ri].any()),weak_safe_pair_retaining_head_available=bool(pair_subject[i,ri].any())))
    summary=[]
    for role in ['fit','dev']:
        for scope in ['essential','secondary','all']:
            for cohort in ['accepted-reference','explicit-contain','unknown-composition','all-nominated-heads']:
                for model in selected:
                    rows=[r for r in predictions if r['role']==role and r['model']==model and r['head_known'] and (scope=='all' or r['essential']==(scope=='essential')) and (cohort=='all-nominated-heads' or (r['reference_known'] if cohort=='accepted-reference' else r['recommendation']=='contain' if cohort=='explicit-contain' else r['recommendation']=='unknown'))]
                    summary.append(dict(role=role,scope=scope,cohort=cohort,model=model,requests=len(rows),families=len({r['family_id'] for r in rows}),weak_cut_requests=sum(r['weak_head_cut'] for r in rows),no_weak_head_retained_requests=sum(r['weak_heads_retained']==0 for r in rows),reference_matches=sum(r['reference_match'] for r in rows)))
    pair_summary=[]
    for role in ['fit','dev']:
        for essential in [True,False]:
            rows=[r for r in pair_rows if r['role']==role and r['essential']==essential and r['head_known']];pair_summary.append(dict(role=role,essential=essential,requests=len(rows),safe_pairs=sum(r['weak_safe_pair_available'] for r in rows),safe_pairs_retaining_head=sum(r['weak_safe_pair_retaining_head_available'] for r in rows)))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},scope='finite nominated head geometry on all FIT/DEV requests, including explicit contain and unknown compositions; never complete face or artistic labels',api_calls=0,test_inference=False,training_admission_changed=False,model_promoted=False));write(a.output/'sources.json',sources);write(a.output/'queries.json',queries);write(a.output/'predictions.json',predictions);write(a.output/'pairs.json',pair_rows);write(a.output/'summary.json',dict(summary=summary,pair_summary=pair_summary,negative_replay_changes=negative_replay_changes,positive_cut_conflicts=positive_cut_conflicts,positive_no_nominated_head_retained=positive_no_head,source_roles=dict(Counter(s['role'] for s in sources)),known_recommendations=dict(Counter(r['recommendation'] for r in queries)),api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(dev_essential=[r for r in summary if r['role']=='dev' and r['scope']=='essential'],pair_summary=pair_summary,known_recommendations=dict(Counter(r['recommendation'] for r in queries)),positive_no_head=positive_no_head)))


if __name__=='__main__':main()
