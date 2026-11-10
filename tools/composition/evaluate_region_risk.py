#!/usr/bin/env python3
"""FIT-selected soft region risk, original quality and actual pair numerical gates."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import torch
from artifact_identity import sha
from distill_attention_linear import write
from person_risk_inputs import load
from region_risk_features import build,features,tensor_sha,head_applicability
from candidate_cut_head import CutHead
from evaluate_person_risk import choose,new_stats,old_stats,STRENGTHS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','parent','miap','unsplash','output']:p.add_argument(name,type=Path)
    p.add_argument('--head-applicability',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    contexts,_,_,ids=load(a.parent,a.miap,a.unsplash);protocol=json.loads((a.fits/'protocol.json').read_text());proposal=json.loads((a.fits/'proposals.json').read_text())
    if proposal['schema']!='region-risk-proposals-v1':raise ValueError('registered proposal schema differs')
    for path,digest in protocol['inputs'].items():
        if sha(Path(path))!=digest:raise ValueError('fit input changed')
    if any(protocol['inputs'].get(path)!=digest for path,digest in ids.items()):raise ValueError('parent context differs')
    active_counts=head_applicability(contexts,proposal['cohorts']) if a.head_applicability else None
    fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if len(fits)!=12:raise ValueError('all12 fits required')
    if json.loads((a.fits/'PROGRESS.json').read_text())['stage']!='complete':raise ValueError('training must be terminal')
    tensors={key:(build(ctx,proposal['cohorts'][key]),build(ctx,proposal['cohorts'][key],permuted=True)) for key,ctx in contexts.items()}
    hashes=json.loads((a.fits/'feature-identities.json').read_text())
    if tensor_sha(tensors['new'][0])!=hashes['real'] or tensor_sha(tensors['new'][1])!=hashes['permuted']:raise ValueError('fit feature arrays do not replay')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={**protocol['inputs'],str(a.fits/'summary.json'):sha(a.fits/'summary.json'),str(a.fits/'proposals.json'):sha(a.fits/'proposals.json')},strengths=STRENGTHS,selection='unchanged FIT essential accepted family cuts with no match/retention loss versus zero',applicability='Lite2 person-positive plus measured head' if a.head_applicability else 'Lite2 person-positive only',measured_head_required=a.head_applicability,active_counts=active_counts,api_calls=0,test_inference=False,model_promoted=False));write(a.output/'feature-identities.json',{key:dict(real=tensor_sha(values[0]),permuted=tensor_sha(values[1])) for key,values in tensors.items()});state=dict(pid=os.getpid(),stage='routing',completed=0,required=12);selected_rows=[];grid=[];numerical=[];predictions=[]
    def progress(**kw):state.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',state)
    def failed():
        if state['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed)
    empty={key:torch.full_like(ctx['data']['eligible'],-1.,dtype=torch.float32).reshape(len(ctx['data']['sources']),11,2,128) for key,ctx in contexts.items()};baseline={key:choose(ctx,empty[key],'all',0.) for key,ctx in contexts.items()};fit_baseline=new_stats(contexts['new'],*baseline['new'],'fit')
    if old_stats(contexts['old'],*baseline['old'])['full_regret']!=.2406301498413086:raise ValueError('original baseline replay differs')
    def report(name,strength,chosen):
        return dict(model=name,strength=strength,new_fit=new_stats(contexts['new'],*chosen['new'],'fit'),new_dev={cohort:new_stats(contexts['new'],*chosen['new'],'dev',cohort) for cohort in ['accepted','contain','unknown','allheads']},new_dev_secondary=new_stats(contexts['new'],*chosen['new'],'dev','allheads',secondary=True),old=old_stats(contexts['old'],*chosen['old']))
    def retain(name,chosen):
        c=contexts['new']['data'];choice,present=chosen;flat=choice.reshape(72,22);offered=present.repeat_interleave(2,1);parent=baseline['new'][0].reshape(72,22);native=contexts['new']['native'].reshape(72,22,128,4)
        for i,s in enumerate(c['sources']):
            if s['role']!='dev':continue
            for q in range(22):
                k=int(flat[i,q]);off=bool(offered[i,q]);old=int(parent[i,q]);predictions.append(dict(model=name,id=s['id'],family_id=s['family_id'],query=q,ratio=c['ratios'][q//2],offered=off,selected=k if off else None,native=native[i,q,k].tolist() if off else None,weak_cut=off and bool(c['negative'][i,q,k]),parent_cut=bool(c['negative'][i,q,old]),reference_match=off and bool(c['positive'][i,q,k]),head_known=bool(contexts['new']['metadata']['head_known'][i,q]),reference_known=bool(contexts['new']['metadata']['reference_known'][i,q])))
    retain('unfiltered',baseline['new']);write(a.output/'baseline.json',report('unfiltered',0.,baseline))
    for entry in fits:
        name=f"{entry['arm']}-seed{entry['seed']}";progress(model=name,completed=len(selected_rows));path=a.fits/name/'selected.pt'
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('selected head changed')
        state_dict=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state_dict['mean'],state_dict['std']);model.load_state_dict(state_dict);model.eval();xs={key:features(ctx['data'],*tensors[key],entry['arm']) for key,ctx in contexts.items()}
        with torch.no_grad():logits={key:model(x).reshape(len(contexts[key]['data']['sources']),11,2,128) for key,x in xs.items()}
        saved=torch.load(a.fits/name/'selected-logits.pt',map_location='cpu',weights_only=False)
        if saved['source_ids']!=[s['id'] for s in contexts['new']['data']['sources']] or saved['checkpoint_sha256']!=sha(path) or not torch.equal(saved['logits'],logits['new'].reshape(72,22,128)):raise ValueError('selected fit logits do not replay exactly')
        best=None;winner=None;selected=None
        with torch.no_grad():
            for strength in STRENGTHS:
                chosen={key:choose(ctx,logits[key],'person',strength) for key,ctx in contexts.items()}
                for key,ctx in contexts.items():
                    if not chosen[key][1].all():raise ValueError('soft risk withdraws crop coverage')
                    if not torch.equal(chosen[key][0][~ctx['person']],baseline[key][0][~ctx['person']]):raise ValueError('person-absent choices change')
                row=report(name,strength,chosen);metric=row['new_fit'];eligible=metric['family_match']>=fit_baseline['family_match'] and metric['family_retained']>=fit_baseline['family_retained'];row['fit_selection_eligible']=eligible;grid.append(row);sort=(metric['family_cut'],-metric['family_match'],-metric['family_retained'],strength)
                if eligible and (best is None or sort<best):best=sort;winner={k:(v[0].clone(),v[1].clone()) for k,v in chosen.items()};selected=row
            if selected is None:raise ValueError('zero must satisfy fixed FIT constraints')
            hard={key:choose(ctx,logits[key],'person') for key,ctx in contexts.items()};selected['hard_diagnostic']=report(name,None,hard)
            for key,ctx in contexts.items():
                c=ctx['data'];x=xs[key];reference=logits[key];compact=model.compact(x).reshape_as(reference);gpu=model.cuda()(x.cuda()).cpu().reshape_as(reference);model.cpu()
                for implementation,actual in [('compact',compact),('cuda',gpu)]:
                    result=dict(model=name,cohort=key,implementation=implementation,max_eligible_logit_difference=float((actual-reference).abs().reshape_as(c['eligible'])[c['eligible']].max()),eligible_cut_flag_changes=int(((actual>=0)!=(reference>=0)).reshape_as(c['eligible'])[c['eligible']].sum()),policies=[])
                    for policy,strength,ref in [('soft',selected['strength'],winner[key]),('hard',None,hard[key])]:
                        chosen,offered=choose(ctx,actual,'person',strength);ref_choice,ref_offered=ref;presence=int((offered!=ref_offered).sum());changed=(chosen!=ref_choice).any(-1)&offered&ref_offered;maximum=0.;violations=presence
                        for i,ri in changed.nonzero().tolist():
                            movement=max(float(((ctx['native'][i,ri,m,chosen[i,ri,m]]-ctx['native'][i,ri,m,ref_choice[i,ri,m]]).abs().double()/ctx['scale'][i,0,0]*128).max()) for m in range(2));maximum=max(maximum,movement);violations+=movement>3
                        result['policies'].append(dict(policy=policy,pair_requests=len(c['sources'])*11,endpoint_changes=int(((chosen!=ref_choice)&offered[...,None]&ref_offered[...,None]).sum()),presence_changes=presence,max_edge_movement_grid=maximum,gate_violations=violations))
                    numerical.append(result)
            retain(name,winner['new']);selected_rows.append(selected);write(a.output/'selected.json',selected_rows);write(a.output/'grid.json',grid);write(a.output/'numerical.json',numerical);write(a.output/'predictions.json',predictions);print(json.dumps(dict(model=name,strength=selected['strength'],new=selected['new_dev']['accepted'],old=selected['old'])),flush=True)
            if any(p['gate_violations'] for r in numerical for p in r['policies']):raise ValueError('original numerical route gate fails')
    write(a.output/'summary.json',dict(selections=selected_rows,models=12,grid_points=len(grid),numerical_rows=len(numerical),numerical_policies=sum(len(r['policies']) for r in numerical),api_calls=0,test_inference=False,model_promoted=False));progress(pid=None,stage='complete',completed=12)


if __name__=='__main__':main()
