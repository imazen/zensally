#!/usr/bin/env python3
"""Fixed-weight source applicability and FIT-selected soft cut-risk controls."""
import argparse
import atexit
import json
import os
from pathlib import Path
import time
import torch
from artifact_identity import sha
from distill_attention_linear import write
from local_rank_distillation import family_weights
from candidate_cut_head import CutHead,features
from candidate_cut_routing import route
from person_risk_policy import adjusted_scores,hard_logits
from person_risk_inputs import load
from fit_nested_pairs import BASELINE

STRENGTHS=[0.,.25,.5,1.,2.,4.,8.]


def new_stats(context,choice,present,role,cohort='accepted',secondary=False):
    c=context['data'];flat=choice.reshape(len(c['sources']),22);offered=present.repeat_interleave(2,1);safe=flat.clamp_min(0);cut=c['negative'].gather(-1,safe[...,None])[...,0]&offered;match=c['positive'].gather(-1,safe[...,None])[...,0]&offered;retained=context['retained'].gather(-1,safe[...,None])[...,0]&offered;m=context['metadata'];mask=m['head_known'].clone();mask&=m['reference_known'] if cohort=='accepted' else m[cohort] if cohort in ['contain','unknown'] else torch.ones_like(mask);mask&=(torch.arange(22)[None]>=18) if secondary else (torch.arange(22)[None]<18);mask&=torch.tensor([s['role']==role for s in c['sources']])[:,None]
    groups={}
    for i,source in enumerate(c['sources']):
        if mask[i].any():groups.setdefault(source['family_id'],[]).append(i)
    means={key:[] for key in ['cut','match','retained']}
    for indices in groups.values():
        denominator=int(mask[indices].sum())
        for key,values in [('cut',cut),('match',match),('retained',retained)]:means[key].append(float((values[indices]&mask[indices]).sum())/denominator)
    count=int(mask.sum());available=int((mask&offered).sum());cuts=int((mask&cut).sum())
    return dict(requests=count,families=len(groups),offered=available,unavailable=count-available,cuts=cuts,matches=int((mask&match).sum()),head_retained=int((mask&retained).sum()),conditional_cut_rate=cuts/available if available else None,**{'family_'+key:sum(value)/len(value) if value else None for key,value in means.items()})


def old_stats(context,choice,present):
    c=context['data'];count=len(c['sources']);flat=choice.reshape(count,22)[:,:18];available=present.repeat_interleave(2,1)[:,:18];t=c['targets'][:,:18];top=t.masked_fill(~c['eligible'][:,:18],-torch.inf).argmax(-1);r=(t.gather(-1,top[...,None])-t.gather(-1,flat.clamp_min(0)[...,None]))[...,0];valid=c['query_valid'][:,:18];known=valid&available;has=known.any(-1);indices=has.nonzero().flatten().tolist();complete=bool(available[valid].all());value=float((((r[has]*known[has]).sum(-1)/known[has].sum(-1))*family_weights(c['sources'],indices)).sum()) if indices else None
    return dict(valid_requests=int(valid.sum()),offered=int(known.sum()),full_regret=value if complete else None,conditional_regret=value,original_full_gate=complete and value<=BASELINE)


def choose(context,logits,applicability,strength=None):
    active=context['person'] if applicability=='person' else torch.ones_like(context['person'])
    if strength is None:return route(context['scores'],context['mask'],hard_logits(logits,active))
    scores=adjusted_scores(context['scores'],logits,active,strength)
    return route(scores,context['mask'],torch.full_like(logits,-1.))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','miap_objects','unsplash_objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);contexts,fits,people,identities=load(a.fits,a.miap_objects,a.unsplash_objects);new=contexts['new'];old=contexts['old'];a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs=identities,classifier_packet_sha256=sha(a.fits/'SHA256SUMS.json'),strengths=STRENGTHS,person_threshold=.3,hard_cut_probability=.5,lambda_selection='new FIT essential accepted-reference family head cuts, subject to no reduction in match/retention from lambda zero; ties prefer match, retention, lower lambda',old_dev_selection=False,new_dev_selection=False,source_roles_changed=False,api_calls=0,test_inference=False,model_promoted=False));write(a.output/'people.json',people);records=[];selections=[];predictions=[];status=dict(pid=os.getpid(),stage='evaluating',completed=0,required=15)
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failure():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failure)
    empty={key:torch.full_like(ctx['data']['eligible'], -1.,dtype=torch.float32).reshape(len(ctx['data']['sources']),11,2,128) for key,ctx in contexts.items()};baseline={key:choose(ctx,empty[key],'all',0.) for key,ctx in contexts.items()};fit_baseline=new_stats(new,*baseline['new'],'fit')
    if old_stats(old,*baseline['old'])['full_regret']!=.2406301498413086:raise ValueError('unfiltered original gate replay differs')
    def report(name,kind,applicability,strength,selected):
        nc,npresent=selected['new'];oc,opresent=selected['old'];return dict(model=name,kind=kind,applicability=applicability,strength=strength,new_fit=new_stats(new,nc,npresent,'fit'),new_dev={cohort:new_stats(new,nc,npresent,'dev',cohort) for cohort in ['accepted','contain','unknown','allheads']},new_dev_secondary=new_stats(new,nc,npresent,'dev','allheads',secondary=True),old=old_stats(old,oc,opresent))
    records.append(report('unfiltered','soft','all',0.,baseline))
    def retain(name,kind,applicability,strength,selected):
        choice,present=selected['new'];c=new['data'];flat=choice.reshape(72,22);offered=present.repeat_interleave(2,1)
        for i,source in enumerate(c['sources']):
            if source['role']!='dev':continue
            for q in range(22):
                k=int(flat[i,q]);exists=bool(offered[i,q]);predictions.append(dict(model=name,kind=kind,applicability=applicability,strength=strength,id=source['id'],family_id=source['family_id'],query=q,offered=exists,selected=k if exists else None,weak_cut=exists and bool(c['negative'][i,q,k]),reference_match=exists and bool(c['positive'][i,q,k])))
    retain('unfiltered','soft','all',0.,baseline)
    for fi,entry in enumerate(fits):
        name=f"{entry['arm']}-seed{entry['seed']}";progress(model=name,completed=fi);path=a.fits/name/'selected.pt'
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('frozen classifier differs')
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);logits={key:model(features(ctx['data'],entry['arm'])).reshape(len(ctx['data']['sources']),11,2,128) for key,ctx in contexts.items()}
        for applicability in ['all','person']:
            selected={key:choose(ctx,logits[key],applicability) for key,ctx in contexts.items()};records.append(report(name,'hard',applicability,None,selected));retain(name,'hard',applicability,None,selected);best=None;chosen=None;chosen_record=None
            for strength in STRENGTHS:
                selected={key:choose(ctx,logits[key],applicability,strength) for key,ctx in contexts.items()}
                for key,ctx in contexts.items():
                    if not selected[key][1].all():raise ValueError('soft risk introduced unavailable pairs')
                    if applicability=='person' and not torch.equal(selected[key][0][~ctx['person']],baseline[key][0][~ctx['person']]):raise ValueError('person-absent source choices changed')
                row=report(name,'soft',applicability,strength,selected);records.append(row);metric=row['new_fit'];eligible=metric['family_match']>=fit_baseline['family_match'] and metric['family_retained']>=fit_baseline['family_retained'];row['fit_selection_eligible']=eligible;key=(metric['family_cut'],-metric['family_match'],-metric['family_retained'],strength)
                if eligible and (best is None or key<best):best=key;chosen={k:(v[0].clone(),v[1].clone()) for k,v in selected.items()};chosen_record=row
            if chosen is None:raise ValueError('lambda zero must meet fixed FIT baseline constraints')
            selections.append(chosen_record);retain(name,'soft',applicability,chosen_record['strength'],chosen)
        write(a.output/'grid.json',records);write(a.output/'selected.json',selections)
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in predictions:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    progress(pid=None,stage='complete',completed=15);write(a.output/'summary.json',dict(selections=selections,grid_configurations=len(records),prediction_rows=len(predictions),people_by_cohort={key:sum(p['applies'] for p in people if p['cohort']==key) for key in contexts},api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(selected=[dict(model=r['model'],applicability=r['applicability'],strength=r['strength'],new=r['new_dev']['accepted'],old=r['old']) for r in selections],people_by_cohort={key:sum(p['applies'] for p in people if p['cohort']==key) for key in contexts})))


if __name__=='__main__':main()
