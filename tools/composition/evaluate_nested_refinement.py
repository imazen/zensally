#!/usr/bin/env python3
"""Bounded exact nested-origin refinement over pinned local head proposals."""
import argparse
import atexit
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import os
import time
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from person_risk_inputs import load
from candidate_cut_head import CutHead,features
from evaluate_person_risk import choose
from refinement_inputs import local_heads,accepted_options,reference_match
from refine_nested_origin import refine
from subject_origin_projection import padded_heads
from subject_policy import face_relation
from distill_attention_linear import write
from crop_corridors import audit


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','parent','faces','pose','original','references','miap_objects','unsplash_objects','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if sha(a.parent/'SHA256SUMS.json')!='8e2456a64197e835ad58a366eb3fd8008ee11b0e629d4c18e7dfd9a4cff44ac2':raise ValueError('frozen person-risk parent differs')
    for name in ['selected.json','people.json','numerical/summary.json']:verify_member(a.parent,name)
    contexts,fits,people,identities=load(a.fits,a.miap_objects,a.unsplash_objects)
    if people!=json.loads((a.parent/'people.json').read_text()):raise ValueError('source applicability differs')
    heads,head_ids=local_heads(contexts,a.faces,a.pose,a.original);identities.update(head_ids);options,nominated,reference_ids,replayed=accepted_options(contexts['new'],a.references);identities.update(reference_ids);selections=json.loads((a.parent/'selected.json').read_text());qualified=json.loads((a.parent/'numerical/summary.json').read_text())['rows'];a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs=identities,parent_manifest_sha256=sha(a.parent/'SHA256SUMS.json'),parents=['hidden-attention-seed31','hidden-attention-seed137','hidden-attention-seed911'],head_families=['faces','faces-pose'],padding=[0,10,25],primary='faces-pose/pad25',reference_query_masks_replayed=replayed,geometry='nearest equal-size head-safe outer; minimal rational child enlargement retaining every outer head; exact continuous interval audit',no_proposals='parent pair retained, unchecked',no_outer='neither endpoint',no_strict_inner='Minimal alone',api_calls=0,test_inference=False,new_gradients=False,model_promoted=False));status=dict(pid=os.getpid(),stage='refining',completed=0,required=18);summaries=[];parents=[];rows=[]
    def progress(**kw):status.update(kw,checked_unix=time.time());write(a.output/'PROGRESS.json',status)
    def failed():
        if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed)
    @lru_cache(maxsize=16384)
    def cached_refine(w,h,outer,inner,ratio,provided):return refine(w,h,list(outer),list(inner),ratio,[list(v) for v in provided])
    padded={}
    for cohort,ctx in contexts.items():
        for source in ctx['data']['sources']:
            for family in ['faces','faces-pose']:
                for padding in [0,10,25]:
                    w,h=source['native_size'];padded[cohort,source['id'],family,padding]=tuple(map(tuple,padded_heads(w,h,heads[cohort][source['id']][family],padding)))
    for seed in [31,137,911]:
        name=f'hidden-attention-seed{seed}';entry=next(r for r in fits if r['arm']=='hidden-attention' and r['seed']==seed);selected=next(r for r in selections if r['model']==name and r['applicability']=='person');check=[r for r in qualified if r['model']==name and r['kind']=='soft' and r['applicability']=='person' and r['implementation']=='full-cuda']
        if len(check)!=2 or any(r['strength']!=selected['strength'] or r['gate_violations'] for r in check):raise ValueError('parent full-neural qualification differs')
        path=a.fits/name/'selected.pt'
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('frozen risk head differs')
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);choices={}
        for cohort,ctx in contexts.items():
            c=ctx['data'];logits=model(features(c,'hidden-attention')).reshape(len(c['sources']),11,2,128);choice,present=choose(ctx,logits,'person',selected['strength'])
            if not present.all():raise ValueError('full-offer parent changed')
            choices[cohort]=choice
            for i,source in enumerate(c['sources']):
                for ri,ratio in enumerate(c['ratios']):
                    pair=[ctx['native'][i,ri,m,int(choice[i,ri,m])].tolist() for m in range(2)];parents.append(dict(seed=seed,cohort=cohort,id=source['id'],role=source['role'],family_id=source['family_id'],ratio=ratio,ratio_index=ri,outer=pair[0],inner=pair[1]))
                    if cohort=='new':
                        for m in range(2):
                            cut=any(face_relation(pair[m],h)=='cut' for h in nominated[source['id']])
                            if cut!=bool(c['negative'][i,ri*2+m,int(choice[i,ri,m])]):raise ValueError('parent nominated-head negative replay differs')
        for family in ['faces','faces-pose']:
            for padding in [0,10,25]:
                policy=f'{family}/pad{padding}';progress(seed=seed,policy=policy,completed=len(summaries));records=[]
                for cohort,ctx in contexts.items():
                    c=ctx['data'];choice=choices[cohort]
                    for i,source in enumerate(c['sources']):
                        w,h=source['native_size'];sid=source['id'];provided=padded[cohort,sid,family,padding]
                        for ri,ratio in enumerate(c['ratios']):
                            parent=[ctx['native'][i,ri,m,int(choice[i,ri,m])].tolist() for m in range(2)]
                            result=cached_refine(w,h,tuple(parent[0]),tuple(parent[1]),ratio,provided) if ctx['person'][i] else dict(outer=parent[0],inner=parent[1],status='person-inapplicable-unchecked',geometry_checked=False,inner_enlarged=False)
                            row=dict(seed=seed,policy=policy,cohort=cohort,id=sid,role=source['role'],family_id=source['family_id'],ratio=ratio,ratio_index=ri,essential=ri<9,person_applies=bool(ctx['person'][i]),provided_heads=len(provided),**result);offered=[result['outer'] is not None,result['inner'] is not None];row['offered']=offered;row['movement_grid']=[max(abs(x-y)*128/s for x,y,s in zip(box,parent[m],[w,h,w,h])) if box is not None else None for m,box in enumerate([result['outer'],result['inner']])]
                            if result['geometry_checked']:
                                for box in [result['outer'],result['inner']]:
                                    if box is not None and any(face_relation(box,list(v))=='cut' for v in provided):raise ValueError('refinement cuts a supplied head')
                                if all(offered) and not audit(result['inner'],result['outer'],w,h,ratio,[list(v) for v in provided])['geometrically_admissible']:raise ValueError('supplied-head continuous interval fails')
                            if cohort=='new':
                                known=source['annotation_valid'] and bool(nominated[sid]);row['head_known']=known;row['reference_known']=[bool(c['reference_valid'][i,ri*2+m].any()) for m in range(2)];row['weak_cut']=[bool(box is not None and any(face_relation(box,v)=='cut' for v in nominated[sid])) for box in [result['outer'],result['inner']]];row['reference_match']=[reference_match(box,options.get((sid,ri*2+m),[]),w,h) for m,box in enumerate([result['outer'],result['inner']])];row['weak_interval_ok']=audit(result['inner'],result['outer'],w,h,ratio,nominated[sid])['geometrically_admissible'] if all(offered) and known else None
                            records.append(row)
                groups=[]
                for cohort,role in [('new','fit'),('new','dev'),('old','dev')]:
                    selected_rows=[r for r in records if r['cohort']==cohort and r['role']==role and r['essential']];counts=dict(requested_pairs=len(selected_rows),outer_offered=sum(r['offered'][0] for r in selected_rows),inner_offered=sum(r['offered'][1] for r in selected_rows),provided_geometry_checked=sum(r['geometry_checked'] for r in selected_rows),inner_enlarged=sum(r['inner_enlarged'] for r in selected_rows),statuses=dict(Counter(r['status'] for r in selected_rows)))
                    if cohort=='new':
                        targets=[(r,m) for r in selected_rows for m in range(2) if r['head_known'] and r['reference_known'][m]];counts.update(accepted_requests=len(targets),accepted_offered=sum(r['offered'][m] for r,m in targets),accepted_weak_cuts=sum(r['weak_cut'][m] for r,m in targets),accepted_matches=sum(r['reference_match'][m] for r,m in targets),known_pairs=sum(r['head_known'] for r in selected_rows),known_offered_pairs=sum(r['head_known'] and all(r['offered']) for r in selected_rows),weak_interval_safe_pairs=sum(r['weak_interval_ok'] is True for r in selected_rows))
                    groups.append(dict(cohort=cohort,role=role,**counts))
                summaries.append(dict(seed=seed,policy=policy,groups=groups));rows.extend(records);write(a.output/'summary.json',dict(policies=summaries,api_calls=0,test_inference=False,model_promoted=False))
    with (a.output/'predictions.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    with (a.output/'parents.jsonl').open('x') as stream:
        for row in parents:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    write(a.output/'sources.json',{cohort:ctx['data']['sources'] for cohort,ctx in contexts.items()});progress(pid=None,stage='complete',completed=18);print(json.dumps([r for r in summaries if r['policy']=='faces-pose/pad25']))


if __name__=='__main__':main()
