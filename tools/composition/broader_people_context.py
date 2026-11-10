"""Fixed broader DEV pair scores with no invented labels or auxiliary features."""
from pathlib import Path
import torch
from artifact_identity import sha
from distill_attention_linear import Student
from pair_relationship_head import PairCorrection


def context(cache,regions,identities):
    if cache['sources']!=regions['sources']:raise ValueError('broader proposal/source identity differs')
    by_hash={digest:Path(path) for path,digest in identities.items()};ep=by_hash['4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'];pp=by_hash['05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede']
    if sha(ep)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b' or sha(pp)!='05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede':raise ValueError('frozen scoring weights differ')
    state=torch.load(ep,map_location='cpu',weights_only=False)['model'];endpoint=Student({k:state['base.'+k] for k in ['weight','bias']},state['teacher'],state['mean'],state['std'],nonlinear=True);endpoint.load_state_dict(state);pair=PairCorrection('additive-linear',state['mean'],state['std']);pair.load_state_dict(torch.load(pp,map_location='cpu',weights_only=False)['model']);endpoint.eval();pair.eval();n=len(cache['sources']);eligible=cache['eligible'].reshape(n,11,2,128);scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in cache['sources']])[:,None,None,:];native=(cache['query_boxes']*scale).round().long().reshape(n,11,2,128,4);outer=native[:,:,0,:,None];inner=native[:,:,1,None,:];oa=(outer[...,2]-outer[...,0])*(outer[...,3]-outer[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);mask=(outer[...,:2]<=inner[...,:2]).all(-1)&(outer[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&eligible[...,0,:,None]&eligible[...,1,None,:]
    if not mask.any((-1,-2)).all():raise ValueError('broader bank lacks a strict pair; retain as explicit feasibility failure')
    with torch.no_grad():hidden=cache['hidden'].reshape(n,11,2,128,96);base=endpoint(hidden,None);scores=pair(hidden,base,eligible)
    return dict(data=cache,native=native,scale=scale,eligible=eligible,mask=mask,scores=scores,person=torch.tensor(regions['person_counts'])>0)
