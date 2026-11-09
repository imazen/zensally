#!/usr/bin/env python3
"""Frozen-score nested-pair policies with explicit endpoint quality coverage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from distill_attention_linear import Student
from nested_pair_selection import select,strict_nested
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import family_weights


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old_hidden','old_rank','new','control','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    expected=[(a.old_hidden,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.old_rank,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3')]
    if any(sha(path)!=identity for path,identity in expected) or sha(a.control/'SHA256SUMS.json')!='941d64dca4c3f0cf5477fdd78ac332dee1a080818c825e44b7c8f17043bb6fc4':raise ValueError('frozen paired-selection inputs differ')
    old=torch.load(a.old_hidden,map_location='cpu',weights_only=False);rank=torch.load(a.old_rank,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);path=a.control/'relationship-rehearsal/replay-contrastive-seed31/selected.pt'
    if sha(path)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b':raise ValueError('selected model differs')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint['model'];model=Student({k:state['base.'+k] for k in ['weight','bias']},state['teacher'],state['mean'],state['std'],nonlinear=True);model.load_state_dict(state);model.eval()
    identity=lambda values:[(s['id'],s['role'],s['family_id'],s['rgb_sha256']) for s in values]
    if identity(old['sources'])!=identity(rank['sources']):raise ValueError('original hidden/rank source roles differ')
    if any(s['role'] not in ['fit','dev'] for s in old['sources']+new['sources']):raise ValueError('TEST must remain closed')
    old_gate=a.control/'relationship-rehearsal-oldgate';members=json.loads((a.control/'SHA256SUMS.json').read_text())
    for name in ['summary.json','replay-contrastive-seed31-predictions.json']:
        if sha(old_gate/name)!=members[str((old_gate/name).relative_to(a.control))]:raise ValueError('original gate evidence changed')
    baseline=json.loads((old_gate/'summary.json').read_text())['baseline'];prior={(r['id'],r['query']):r['selected'] for r in json.loads((old_gate/'replay-contrastive-seed31-predictions.json').read_text())}
    policies=['independent','outer-first','joint-sum'];pairs=[];endpoints=[];raw_changes=[]
    a.output.mkdir(parents=True,exist_ok=False)
    for cohort,data,geometry in [('old',old,rank),('new',new,new)]:
        sources=data['sources'];all_scores=torch.empty_like(data['initial_scores'])
        for role in ['fit','dev']:
            indices=[i for i,s in enumerate(sources) if s['role']==role];all_scores[indices]=model(data['hidden'][indices],None)
        for i,source in enumerate(sources):
            w,h=source['native_size'];scale=torch.tensor([w,h,w,h]);values=all_scores[i].tolist();eligible=data['eligible'][i].tolist()
            for ri,ratio in enumerate(data['ratios']):
                p,r=map(int,ratio.split(':'));q=ri*2
                outer=(geometry['boxes'][i,ri]*scale).round().int().tolist() if cohort=='old' else (data['query_boxes'][i,q]*scale).round().int().tolist()
                inner=(geometry['boxes'][i,ri]*scale).round().int().tolist() if cohort=='old' else (data['query_boxes'][i,q+1]*scale).round().int().tolist()
                for boxes,mask in [(outer,eligible[q]),(inner,eligible[q+1])]:
                    for box,active in zip(boxes,mask):
                        if active and (not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h) or (box[2]-box[0])*r!=(box[3]-box[1])*p):raise ValueError('eligible native geometry/aspect differs')
                raw=select(outer,inner,values[q],values[q+1],eligible[q],eligible[q+1],'independent')
                for query,index in [(q,raw[0]),(q+1,raw[1])]:
                    if cohort=='old' and source['role']=='dev' and index!=prior[source['id'],query]:raw_changes.append(dict(id=source['id'],query=query,prior=prior[source['id'],query],cached=index))
                for policy in policies:
                    oi,ii=select(outer,inner,values[q],values[q+1],eligible[q],eligible[q+1],policy)
                    offered=ii is not None;strict=offered and strict_nested(inner[ii],outer[oi])
                    if policy!='independent' and offered and not strict:raise ValueError('structured pair is not strictly nested')
                    if policy=='outer-first' and oi!=raw[0]:raise ValueError('outer-first moved raw Minimal')
                    if policy=='outer-first' and strict_nested(inner[raw[1]],outer[raw[0]]) and ii!=raw[1]:raise ValueError('already nested raw pair changed')
                    if outer[oi][2]-outer[oi][0]!=p*min(w//p,h//r) or outer[oi][3]-outer[oi][1]!=r*min(w//p,h//r):raise ValueError('Minimal outer lost maximum area')
                    pairs.append(dict(cohort=cohort,id=source['id'],family_id=source['family_id'],role=source['role'],ratio=ratio,ratio_index=ri,essential=ri<9,policy=policy,raw_outer=raw[0],raw_inner=raw[1],outer=oi,inner=ii,pair_offered=offered,strict_pair=strict,minimal_changed=oi!=raw[0],maximal_changed=ii!=raw[1],outer_box=outer[oi],inner_box=inner[ii] if offered else None))
                    for query,index,box in [(q,oi,outer[oi]),(q+1,ii,inner[ii] if offered else None)]:
                        available=index is not None;record=dict(cohort=cohort,id=source['id'],family_id=source['family_id'],role=source['role'],query=query,ratio=ratio,mode='minimal' if query%2==0 else 'maximal',essential=query<18,policy=policy,selected=index,offered=available,native_box=box)
                        if cohort=='old':
                            target=data['targets'][i,query];teacher=int(target.masked_fill(~data['eligible'][i,query],-torch.inf).argmax());record.update(valid=bool(data['query_valid'][i,query]),teacher_regret=float(target[teacher]-target[index]) if available else None)
                        else:record.update(valid=bool(data['reference_valid'][i,query].any()),cut=available and bool(data['negative'][i,query,index]),match=available and bool(data['positive'][i,query,index]))
                        endpoints.append(record)
        del all_scores
    if raw_changes:raise ValueError('old raw numerical replay differs: '+json.dumps(raw_changes))
    summary=[]
    for cohort in ['old','new']:
        for role in ['fit','dev']:
            for scope in ['all','essential']:
                for policy in policies:
                    selected=[r for r in pairs if r['cohort']==cohort and r['role']==role and r['policy']==policy and (scope=='all' or r['essential'])]
                    summary.append(dict(cohort=cohort,role=role,scope=scope,policy=policy,requests=len(selected),offered_pairs=sum(r['pair_offered'] for r in selected),strict_pairs=sum(r['strict_pair'] for r in selected),minimal_changes=sum(r['minimal_changed'] for r in selected),maximal_changes=sum(r['maximal_changed'] for r in selected)))
    quality=[];old_dev=[i for i,s in enumerate(old['sources']) if s['role']=='dev'];old_lookup={s['id']:i for i,s in enumerate(old['sources'])}
    for policy in policies:
        for scope,qs in [('all',range(22)),('essential',range(18)),('secondary',range(18,22))]:
            selected=[r for r in endpoints if r['cohort']=='old' and r['role']=='dev' and r['policy']==policy and r['query'] in qs and r['valid']];groups=defaultdict(list)
            for row in selected:groups[row['id']].append(row)
            complete=all(r['offered'] for r in selected);present=[sid for sid,rows in groups.items() if any(r['offered'] for r in rows)];indices=[old_lookup[sid] for sid in present];weights=family_weights(old['sources'],indices);means=[];raw_means=[]
            raw={(r['id'],r['query']):r for r in endpoints if r['cohort']=='old' and r['policy']=='independent' and r['valid']}
            for sid in present:
                offered=[r for r in groups[sid] if r['offered']];means.append(torch.tensor([r['teacher_regret'] for r in offered]).mean());raw_means.append(torch.tensor([raw[sid,r['query']]['teacher_regret'] for r in offered]).mean())
            conditional=float((torch.stack(means)*weights).sum()) if means else None;matched=float((torch.stack(raw_means)*weights).sum()) if means else None
            quality.append(dict(policy=policy,scope=scope,valid_requests=len(selected),offered_requests=sum(r['offered'] for r in selected),conditional_regret=conditional,raw_on_same_offered_regret=matched,full_regret=conditional if complete else None,full_nonregression_proven=complete and conditional<=baseline[scope]['teacher_regret'],baseline_regret=baseline[scope]['teacher_regret']))
    weak=[]
    for policy in policies:
        selected=[r for r in endpoints if r['cohort']=='new' and r['role']=='dev' and r['policy']==policy and r['essential'] and r['valid']];groups=defaultdict(list)
        for row in selected:groups[row['family_id']].append(row)
        weak.append(dict(policy=policy,requests=len(selected),families=len(groups),offered_requests=sum(r['offered'] for r in selected),cut_requests=sum(r['cut'] for r in selected),matched_requests=sum(r['match'] for r in selected),**{key:sum(sum(r[key] for r in group)/len(group) for group in groups.values())/len(groups) for key in ['offered','cut','match']}))
    write=lambda name,value:(a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    write('protocol.json',dict(inputs={str(path):identity for path,identity in expected},checkpoint_sha256=sha(path),policies=policies,pair_sum='binary64 addition of binary32 scores; index ties ascending',strictly_smaller=True,missing='Minimal retained, Maximal unavailable; never zero regret',no_annotation_filters_in_selection=True,api_calls=0,test_inference=False,model_promoted=False))
    write('pairs.json',pairs);write('endpoints.json',endpoints);write('sources.json',dict(old=rank['sources'],new=new['sources']));write('summary.json',dict(pair_summary=summary,old_quality=quality,new_weak=weak,old_raw_choice_changes=len(raw_changes),api_calls=0,test_inference=False,model_promoted=False,face_coverage_certified=False))
    print(json.dumps(dict(pair_summary=[r for r in summary if r['role']=='dev' and r['scope']=='essential'],old_quality=[r for r in quality if r['scope']=='essential'],new_weak=weak)))


if __name__=='__main__':main()
