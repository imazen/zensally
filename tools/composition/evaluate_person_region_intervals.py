#!/usr/bin/env python3
"""Exact automatic person-region constraints, independent weak head diagnostics."""
import argparse
import atexit
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import write
from person_risk_policy import person_count
from roi_geometry import person_views
from refine_nested_origin import refine
from crop_corridors import audit
from fit_nested_crop_head import audited_predictions
from refinement_inputs import reference_match
from subject_policy import face_relation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','direct','objects','baseline','pca','raw','output']:p.add_argument(key,type=Path)
    p.add_argument('--regions-json',type=Path);p.add_argument('--regions-sha');a=p.parse_args();torch.set_num_threads(2)
    if (a.regions_json is None)!=(a.regions_sha is None):raise ValueError('explicit region hash required')
    fixed=[(a.cache,'3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4'),(a.direct,'2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472'),(a.objects/'manifest.json','eb83a50e5c6760727b495d4d2e28b1f9d8c5fa23b1f844b3fc65c9beec0d1fa7'),(a.objects/'full/tf-efficientdet_lite2/results.json','e5cf955399185392c7413aa8620ddb5113aa2f65a1bd6a31f75a3f40a6bbd3a9'),(a.baseline/'SHA256SUMS.json','318f88d1d3259433ead7d424f082a5ec63cb9908d1578e242968449286f910d5'),(a.pca/'SHA256SUMS.json','a16fbdd8f9dfbf24bf62c459621daf3a87c1a4b1bade79a4317ae44372b4190e'),(a.raw/'SHA256SUMS.json','48be1aa863bdabaf26c7a17cc5eee4e69362c7983080ccc38e9bc1e0a269d299')]
    if any(sha(path)!=digest for path,digest in fixed):raise ValueError('registered inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);d=torch.load(a.direct,map_location='cpu',weights_only=False)
    manifest=json.loads((a.objects/'manifest.json').read_text());images={r['id']:r for r in manifest['images']};detected={r['id']:r['result']['detections'] for r in json.loads((a.objects/'full/tf-efficientdet_lite2/results.json').read_text())['rows']}
    if len(c['sources'])!=72 or len(c['ratios'])!=11 or manifest['confidence_threshold']!=.3 or manifest['max_results']!=50:raise ValueError('fixed source/detector settings differ')
    provided={};source_rows=[]
    for source in c['sources']:
        sid=source['id'];count=person_count(source,images[sid],detected[sid]);views={'person':[],'upper':[]}
        for detection in detected[sid]:
            if not any(v['name']=='person' and v['score']>=.3 for v in detection['categories']):continue
            for name,box in person_views(detection['xywh'],source['working_size'],source['native_size']):views[name].append(box)
        if any(len(v)!=count for v in views.values()):raise ValueError('source/person count differs')
        for name,boxes in views.items():provided[sid,name]=tuple(sorted(set(map(tuple,boxes))))
        source_rows.append(dict(id=sid,source=source,person_count=count,regions={name:[list(b) for b in provided[sid,name]] for name in views}))
    policies=['upper','person']
    if a.regions_json is not None:
        if sha(a.regions_json)!=a.regions_sha:raise ValueError('adaptive region content differs')
        extra=json.loads(a.regions_json.read_text())
        if extra['schema']!='source-head-body-fallback-v1' or extra['sources']!=c['sources'] or set(extra['variants'])!={'face-body','face-pose-body'}:raise ValueError('adaptive source/region schema differs')
        policies=['face-body','face-pose-body']
        for name in policies:
            if set(extra['variants'][name])!={s['id'] for s in c['sources']}:raise ValueError('adaptive sources differ')
            for sid,boxes in extra['variants'][name].items():provided[sid,name]=tuple(sorted(set(map(tuple,boxes))))
        for row in source_rows:row['regions']={name:[list(b) for b in provided[row['id'],name]] for name in policies}
        fixed.append((a.regions_json,a.regions_sha))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in fixed},policies=policies,primary='face-pose-body' if a.regions_json is not None else 'upper',parents=['geometry','person-all-pca8','person-largest-pca8','person-all-raw','person-largest-raw'],api_calls=0,new_inference=False,new_gradients=False,test_inference=False,model_promoted=False))
    write(a.output/'sources.json',source_rows)
    state=dict(pid=os.getpid(),stage='refining',completed=0,required=10);reports=[]
    def progress(**kw):state.update(kw);write(a.output/'PROGRESS.json',state)
    def failed():
        if state['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
    atexit.register(failed)
    @lru_cache(maxsize=8192)
    def solve(w,h,outer,inner,ratio,regions):return refine(w,h,list(outer),list(inner),ratio,[list(b) for b in regions])
    for parent,root in [('geometry',a.baseline),('person-all-pca8',a.pca/'fits'),('person-largest-pca8',a.pca/'fits'),('person-all-raw',a.raw/'fits'),('person-largest-raw',a.raw/'fits')]:
        packet=a.baseline if parent=='geometry' else a.pca if parent.endswith('pca8') else a.raw
        member=(parent+'-predictions.json') if parent=='geometry' else 'fits/'+parent+'-predictions.json';verify_member(packet,member)
        frozen=json.loads((root/(parent+'-predictions.json')).read_text());lookup={(r['id'],r['query']):r for r in frozen}
        native=torch.tensor([[[lookup[s['id'],ri*2+m]['native'] for m in range(2)] for ri in range(11)] for s in c['sources']],dtype=torch.float64)
        _,replay=audited_predictions(native,c,d)
        if replay!=frozen:raise ValueError('parent diagnostic rows do not replay exactly')
        for policy in policies:
            progress(parent=parent,policy=policy);rows=[];pair_rows=[]
            for i,s in enumerate(c['sources']):
                sid=s['id'];w,h=s['native_size'];heads=c['heads'][sid];regions=provided[sid,policy]
                for ri,ratio in enumerate(c['ratios']):
                    before=[lookup[sid,ri*2+m]['native'] for m in range(2)];result=solve(w,h,tuple(before[0]),tuple(before[1]),ratio,regions);boxes=[result['outer'],result['inner']]
                    if all(box is not None for box in boxes):
                        if not audit(boxes[1],boxes[0],w,h,ratio,[list(b) for b in regions])['geometrically_admissible'] and regions:raise ValueError('supplied-region continuous guard fails')
                        if (boxes[1][2]-boxes[1][0])>=(boxes[0][2]-boxes[0][0]):raise ValueError('strict nesting changed')
                    if boxes[0] is not None and [boxes[0][2]-boxes[0][0],boxes[0][3]-boxes[0][1]]!=[before[0][2]-before[0][0],before[0][3]-before[0][1]]:raise ValueError('maximum-area Minimal changed')
                    head_known=s['annotation_valid'] and bool(heads)
                    pair_rows.append(dict(id=sid,role=s['role'],ratio=ratio,ratio_index=ri,provided_regions=len(regions),**result,weak_interval_safe=audit(boxes[1],boxes[0],w,h,ratio,heads)['geometrically_admissible'] if all(b is not None for b in boxes) and head_known else None))
                    for m,box in enumerate(boxes):
                        old=lookup[sid,ri*2+m];offered=box is not None
                        if offered and regions and any(face_relation(box,list(b))=='cut' for b in regions):raise ValueError('offered crop cuts supplied region')
                        rows.append(dict(old,native=box,offered=offered,reference_match=reference_match(box,c['reference_options'].get((sid,ri*2+m),[]),w,h),weak_cut=bool(offered and any(face_relation(box,b)=='cut' for b in heads)),head_retained=bool(offered and any(face_relation(box,b)=='contained' for b in heads)),interval_safe=pair_rows[-1]['weak_interval_safe'],parent_cut=old['weak_cut'],parent_reference_match=old['reference_match'],movement_grid=max(abs(x-y)*128/z for x,y,z in zip(box,before[m],[w,h,w,h])) if offered else None))
            groups=[]
            for role in ['fit','dev']:
                for secondary in [False,True]:
                    chosen=[r for r in rows if r['role']==role and (r['query']>=18)==secondary];known=[r for r in chosen if r['head_known']];accepted=[r for r in known if r['reference_known']];pairs=[r for r in pair_rows if r['role']==role and (r['ratio_index']>=9)==secondary]
                    groups.append(dict(role=role,secondary=secondary,requested=len(chosen),offered=sum(r['offered'] for r in chosen),head_known=len(known),all_known_cuts=sum(r['weak_cut'] for r in known),all_known_heads_retained=sum(r['head_retained'] for r in known),accepted=len(accepted),accepted_offered=sum(r['offered'] for r in accepted),accepted_cuts=sum(r['weak_cut'] for r in accepted),accepted_heads_retained=sum(r['head_retained'] for r in accepted),accepted_matches=sum(r['reference_match'] for r in accepted),common_offer_repairs=sum(r['offered'] and r['parent_cut'] and not r['weak_cut'] for r in accepted),common_offer_new_cuts=sum(r['offered'] and not r['parent_cut'] and r['weak_cut'] for r in accepted),lost_accepted_matches=sum(r['parent_reference_match'] and not r['reference_match'] for r in accepted),teacher_contain_requests=sum(r['teacher_contain'] for r in chosen),teacher_contain_emitted=sum(r['teacher_contain'] and r['offered'] for r in chosen),statuses=dict(Counter(r['status'] for r in pairs)),inner_enlarged=sum(r['inner_enlarged'] for r in pairs),weak_interval_safe_pairs=sum(r['weak_interval_safe'] is True for r in pairs)))
            out=a.output/(parent+'-'+policy);out.mkdir();write(out/'predictions.json',rows);write(out/'pairs.json',pair_rows);report=dict(parent=parent,policy=policy,groups=groups);write(out/'summary.json',report);reports.append(report);write(a.output/'summary.json',reports);progress(completed=len(reports));print(json.dumps(dict(parent=parent,policy=policy,dev=next(g for g in groups if g['role']=='dev' and not g['secondary']))),flush=True)
            if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>16*1024**2:raise ValueError('registered packet budget exceeded')
    progress(pid=None,stage='complete',completed=10)


if __name__=='__main__':main()
