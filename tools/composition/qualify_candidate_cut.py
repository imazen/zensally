#!/usr/bin/env python3
"""Cached cut-head CPU/compact/CUDA replay of both actual paired routing policies."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from candidate_cut_head import CutHead,features
from candidate_cut_routing import route
from distill_attention_linear import Student,write
from pair_relationship_head import PairCorrection
from nested_pair_training import compatibility


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','new','old','rank','auxiliary','endpoint','pairs','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected=[(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.old,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.auxiliary,'848f2647921697f112f106cf13b11b016f24aa13b76b6621f5c77d4728f6eddc'),(a.endpoint,'4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'),(a.pairs/'additive-linear-seed137/selected.pt','05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered numerical inputs differ')
    new=torch.load(a.new,map_location='cpu',weights_only=False);old=torch.load(a.old,map_location='cpu',weights_only=False);rank=torch.load(a.rank,map_location='cpu',weights_only=False);aux=torch.load(a.auxiliary,map_location='cpu',weights_only=False);ix=aux['indices']
    if aux['sources']!=[old['sources'][i] for i in ix] or aux['rank_sha256']!=sha(a.rank) or any(old['sources'][i]['role']!='dev' for i in ix):raise ValueError('original DEV auxiliary identity differs')
    original=dict(sources=aux['sources'],ratios=old['ratios'],hidden=old['hidden'][ix],eligible=old['eligible'][ix],query_boxes=rank['boxes'][ix].repeat_interleave(2,1),extras=aux['extras'],permuted_extras=aux['permuted_extras']);s=torch.load(a.endpoint,map_location='cpu',weights_only=False)['model'];endpoint=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);endpoint.load_state_dict(s);pair=PairCorrection('additive-linear',s['mean'],s['std']);pair.load_state_dict(torch.load(a.pairs/'additive-linear-seed137/selected.pt',weights_only=False)['model']);fits=json.loads((a.fits/'summary.json').read_text())['fits'];reports=[]
    for cohort,c in [('new',new),('old-dev',original)]:
        count=len(c['sources']);scale=torch.tensor([[*v['native_size'],*v['native_size']] for v in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().long().reshape(count,11,2,128,4);e=c['eligible'].reshape(count,11,2,128);o=native[:,:,0,:,None,:];inner=native[:,:,1,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);mask=(o[...,:2]<=inner[...,:2]).all(-1)&(o[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&e[...,0,:,None]&e[...,1,None,:];h=c['hidden'].reshape(count,11,2,128,96);base=endpoint(h,None);scores={'joint':base[...,0,:,None].double()+base[...,1,None,:].double(),'corrected':pair(h,base,e)}
        for entry in fits:
            name=f"{entry['arm']}-seed{entry['seed']}";path=a.fits/name/'selected.pt'
            if sha(path)!=entry['checkpoint_sha256']:raise ValueError('selected classifier changed')
            state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);x=features(c,entry['arm']);reference=model(x);compact=model.compact(x);gpu=model.cuda()(x.cuda()).cpu()
            for implementation,logits in [('compact',compact),('cuda',gpu)]:
                numeric=dict(cohort=cohort,model=name,implementation=implementation,candidate_logits=int(c['eligible'].sum()),max_absolute_logit_difference=float((logits-reference).abs().max()),eligible_cut_flag_changes=int(((logits>=0)!=(reference>=0))[c['eligible']].sum()),policies=[])
                for policy,value in scores.items():
                    ref,present=route(value,mask,reference.reshape(count,11,2,128));actual,offered=route(value,mask,logits.reshape(count,11,2,128));presence=int((present!=offered).sum());changed=(actual!=ref).any(-1)&present&offered;edge=0.;violations=presence
                    for j,ri in changed.nonzero().tolist():
                        movement=max(float(((native[j,ri,m,actual[j,ri,m]]-native[j,ri,m,ref[j,ri,m]]).abs().float()/scale[j,0,0]*128).max()) for m in range(2));edge=max(edge,movement);violations+=movement>3
                    numeric['policies'].append(dict(policy=policy,pair_requests=count*11,endpoint_changes=int(((actual!=ref)&present[...,None]&offered[...,None]).sum()),presence_changes=presence,max_edge_movement_grid=edge,gate_violations=violations))
                reports.append(numeric)
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'summary.json',dict(inputs={str(path):digest for path,digest in expected},rows=reports,scope='cached measured candidate inputs through classifier and two fixed score routing policies; excludes encoder, preprocessing and teacher inference',api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(rows=len(reports),max_difference=max(r['max_absolute_logit_difference'] for r in reports),cut_flag_changes=sum(r['eligible_cut_flag_changes'] for r in reports),endpoint_changes=sum(p['endpoint_changes'] for r in reports for p in r['policies']),presence_changes=sum(p['presence_changes'] for r in reports for p in r['policies']),gate_violations=sum(p['gate_violations'] for r in reports for p in r['policies']))))
    if any(p['gate_violations'] for r in reports for p in r['policies']):raise SystemExit('numerical route gate failed')


if __name__=='__main__':main()
