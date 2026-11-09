#!/usr/bin/env python3
"""Describe guarded pair quality, lost offers and unmeasured auxiliary inputs."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from candidate_cut_head import CutHead,features,geometry_features
from candidate_cut_routing import route
from pair_relationship_head import PairCorrection
from nested_pair_training import compatibility
from local_rank_distillation import family_weights
from fit_nested_pairs import BASELINE


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','cache','old_hidden','old_rank','endpoint','pairs','coverage','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    expected=[(a.cache,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.old_hidden,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.old_rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.endpoint,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'),(a.pairs/'SHA256SUMS.json','698972129157fa6ffce1c748f94bacac12a42b703ddea7c300b5445743b93020'),(a.coverage/'SHA256SUMS.json','160d693d91bae8dd858036c0bd198088af26be847122e50bdad8fa2ec336a34a')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('frozen routing inputs differ')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);old=torch.load(a.old_hidden,map_location='cpu',weights_only=False);rank=torch.load(a.old_rank,map_location='cpu',weights_only=False);s=torch.load(a.endpoint,map_location='cpu',weights_only=False)['model'];endpoint=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);endpoint.load_state_dict(s);endpoint.eval();pair=PairCorrection('additive-linear',s['mean'],s['std']);pair_path=a.pairs/'additive-linear-seed137/selected.pt'
    if sha(pair_path)!='05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede':raise ValueError('fixed additive control differs')
    pair.load_state_dict(torch.load(pair_path,map_location='cpu',weights_only=False)['model']);metadata={(r['id'],r['query']):r for r in json.loads((a.coverage/'queries.json').read_text())};fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if len(fits)!=15:raise ValueError('registered fifteen classifier fits required')
    rows=[];summaries=[];old_quality=[];replay=[]
    for cohort,c in [('new',cache),('old',old)]:
        indices=[i for i,v in enumerate(c['sources']) if v['role']=='dev'];count=len(indices);hidden=c['hidden'][indices].reshape(count,11,2,128,96);base=endpoint(hidden,None);e=c['eligible'][indices].reshape(count,11,2,128)
        if count!=(24 if cohort=='new' else 25):raise ValueError('fixed DEV coverage differs')
        if cohort=='old':
            scale=torch.tensor([[*v['native_size'],*v['native_size']] for v in rank['sources']])[:,None,None,:];native=(rank['boxes']*scale).round().long();mask=compatibility(native[indices],e);geometry=geometry_features(rank['sources'],rank['ratios'],rank['boxes'].repeat_interleave(2,1));data=dict(c,query_boxes=rank['boxes'].repeat_interleave(2,1))
        else:
            scale=torch.tensor([[*v['native_size'],*v['native_size']] for v in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().long();o=native[indices,::2,:,None,:];inner=native[indices,1::2,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);mask=(o[...,:2]<=inner[...,:2]).all(-1)&(o[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&e[...,0,:,None]&e[...,1,None,:];geometry=None;data=c
        policies={'joint':base[...,0,:,None].double()+base[...,1,None,:].double(),'corrected':pair(hidden,base,e)}
        for fit in [None,*fits]:
            name='unfiltered' if fit is None else f"{fit['arm']}-seed{fit['seed']}"
            if cohort=='old' and fit is not None and fit['arm'] not in ['geometry','hidden']:
                for policy in policies:old_quality.append(dict(model=name,policy=policy,measured=False,reason='original pose/gaze/scanpath inputs not present; no zero fabrication'))
                continue
            logits=torch.full((count,11,2,128),-1.)
            if fit is not None:
                path=a.fits/name/'selected.pt'
                if sha(path)!=fit['checkpoint_sha256']:raise ValueError('selected classifier changed')
                state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);x=features(data,fit['arm'],geometry)[indices];logits=model(x).reshape(count,11,2,128)
                if cohort=='new':
                    saved=torch.load(a.fits/name/'selected-logits.pt',map_location='cpu',weights_only=False)
                    if saved['source_ids']!=[v['id'] for v in c['sources']] or saved['roles']!=[v['role'] for v in c['sources']] or saved['checkpoint_sha256']!=sha(path) or not torch.equal(logits.reshape(count,22,128),saved['logits'][indices]):raise ValueError('selected classifier logit replay differs')
                    replay.append(dict(model=name,selected_logits_exact=True))
            for policy,scores in policies.items():
                chosen,present=route(scores,mask,logits);flat=chosen.reshape(count,22);available=present.repeat_interleave(2,1)
                if cohort=='new':
                    records=[]
                    for j,i in enumerate(indices):
                        for q in range(22):
                            k=int(flat[j,q]);meta=metadata[c['sources'][i]['id'],q];offered=bool(available[j,q]);records.append(dict(model=name,policy=policy,id=c['sources'][i]['id'],family_id=c['sources'][i]['family_id'],query=q,essential=q<18,head_known=meta['head_known'],reference_known=meta['reference_known'],recommendation=meta['recommendation'],offered=offered,selected=k if offered else None,weak_cut=offered and bool(c['negative'][i,q,k]),reference_match=offered and bool(c['positive'][i,q,k])))
                    rows.extend(records)
                    for scope in ['essential','secondary','all']:
                        for group in ['accepted-reference','explicit-contain','unknown-composition','all-nominated-heads']:
                            selected=[r for r in records if r['head_known'] and (scope=='all' or r['essential']==(scope=='essential')) and (group=='all-nominated-heads' or (r['reference_known'] if group=='accepted-reference' else r['recommendation']=='contain' if group=='explicit-contain' else r['recommendation']=='unknown'))];offered=sum(r['offered'] for r in selected);cuts=sum(r['weak_cut'] for r in selected);summaries.append(dict(model=name,policy=policy,scope=scope,cohort=group,requests=len(selected),offered=offered,unavailable=len(selected)-offered,cuts=cuts,cut_rate_among_offered=cuts/offered if offered else None,matches=sum(r['reference_match'] for r in selected)))
                else:
                    t=c['targets'][indices,:18];eligible=c['eligible'][indices,:18];top=t.masked_fill(~eligible,-torch.inf).argmax(-1);valid=c['query_valid'][indices,:18];safe_choice=flat[:,:18].clamp_min(0);regret=(t.gather(-1,top[...,None])-t.gather(-1,safe_choice[...,None]))[...,0];known_offered=valid&available[:,:18];complete=bool(available[:,:18][valid].all());has=known_offered.any(-1);position=has.nonzero().flatten().tolist();weights=family_weights(c['sources'],[indices[j] for j in position]);conditional=float((((regret[has]*known_offered[has]).sum(-1)/known_offered[has].sum(-1))*weights).sum()) if position else None;old_quality.append(dict(model=name,policy=policy,measured=True,valid_requests=int(valid.sum()),offered_requests=int(known_offered.sum()),full_regret=conditional if complete else None,conditional_regret=conditional,full_original_gate=complete and conditional<=BASELINE))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},probability_threshold=.5,score_policies=['joint','corrected fixed additive137'],missing='both endpoints unavailable; no fallback or invented regret',selection_affected=False,api_calls=0,test_inference=False,model_promoted=False));write(a.output/'summary.json',dict(new=summaries,old=old_quality,logit_replay=replay,api_calls=0,test_inference=False,model_promoted=False));write(a.output/'predictions.json',rows);print(json.dumps(dict(new_essential=[r for r in summaries if r['scope']=='essential' and r['cohort']=='accepted-reference'],old=old_quality)))


if __name__=='__main__':main()
