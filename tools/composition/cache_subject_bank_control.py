#!/usr/bin/env python3
"""Build an explicit bank-only weak-label control without changing parent caches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from subject_candidate_bank import generate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import audit,pixels,project
from subject_policy import face_relation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','reference_cache','rank_cache','output']:p.add_argument(key,type=Path)
    p.add_argument('--bank',choices=['old-broad-maximal','hybrid'],required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.rank_cache)!='420b7fad28e64af4292f1c155d0447abd106b0fe190bbd07bb88d750d1df804d':raise ValueError('frozen native parent differs')
    parent=torch.load(a.reference_cache,map_location='cpu',weights_only=False)
    original=torch.load(a.rank_cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(original['sources'])}
    if parent['schema']!='subject-choice-fixed-bank-v1' or parent['protocol']['reference_tolerance_grid']!=3:raise ValueError('frozen subject references required')
    box_rows=[];eligible_rows=[];positive_rows=[];negative_rows=[];counts=Counter();queries=[]
    for i,source in enumerate(parent['sources']):
        pi=lookup[source['id']];w,h=source['native_size'];scale=np.array([w,h,w,h],np.float64)
        zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text()) if source['annotation_valid'] else None
        requests={(r['ratio'],r['mode']):r for r in zones['crops']} if zones else {}
        faces=[pixels(r['box'],w,h,outward=True) for r in zones['rois'] if r['kind']=='face_head'] if zones else []
        boxes=np.zeros((22,128,4),np.float32);eligible=np.zeros((22,128),bool);positive=np.zeros_like(eligible);negative=np.zeros_like(eligible)
        for q in range(22):
            ratio=parent['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal';p,r=map(int,ratio.split(':'))
            dense=a.bank=='hybrid' and mode=='minimal'
            native=generate(w,h,(p,r),mode) if dense else original['native_boxes'][pi][q//2]
            n=len(native);raw=np.asarray(native,np.float64)
            if not(0<n<=128):raise ValueError('bounded bank required')
            boxes[q,:n]=raw/scale;boxes[q,n:]=boxes[q,n-1]
            eligible[q,:n]=True if dense or mode=='maximal' else parent['eligible'][i,q,:n].numpy()
            if faces:negative[q,:n]=[any(face_relation(list(b),face)=='cut' for face in faces) for b in native]
            negative[q]&=eligible[q]
            request=requests.get((ratio,mode));good=[]
            if request:
                for option in request['options'] if request['action']!='contain' else []:
                    raw_target=pixels(option['box'],w,h);exact=project(option['box'],w,h,ratio,mode)
                    args=(zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)
                    if option['face_policy_check']!='whole_faces_or_excluded' or not audit(raw_target,*args,outward=False)['geometry_compatible'] or not audit(exact,*args)['geometry_compatible'] or max(np.abs(np.asarray(exact)-raw_target)/scale*128)>3:continue
                    good.append((np.asarray(exact)/scale,option))
            if len(good)!=int(parent['reference_valid'][i,q].sum()):raise ValueError('accepted options changed')
            for slot,(target,option) in enumerate(good):
                if not np.array_equal(target.astype(np.float32),parent['references'][i,q,slot].numpy()):raise ValueError('accepted coordinates changed')
                close=(np.abs(raw/scale-target)*128).max(-1)<=3
                safe=np.array([audit(list(b),zones['rois'],option['selected_subject_ids'],option['excluded_subject_ids'],w,h)['geometry_compatible'] for b in native])
                positive[q,:n]|=close&safe&eligible[q,:n]
            if np.any(positive[q]&negative[q]):raise ValueError('positive/negative conflict')
            if good:
                counts['accepted_queries']+=1;counts['covered_queries']+=bool(positive[q].any())
                counts['contrastive_queries']+=bool(positive[q].any() and negative[q].any())
                queries.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],query=q,ratio=ratio,mode=mode,
                    covered=bool(positive[q].any()),contrastive=bool(positive[q].any() and negative[q].any()),eligible=int(eligible[q].sum())))
        box_rows.append(boxes);eligible_rows.append(eligible);positive_rows.append(positive);negative_rows.append(negative)
    output=dict(parent,schema='subject-choice-query-bank-v1',query_boxes=torch.from_numpy(np.array(box_rows)),
        eligible=torch.from_numpy(np.array(eligible_rows)),positive=torch.from_numpy(np.array(positive_rows)),negative=torch.from_numpy(np.array(negative_rows)))
    output['protocol']=dict(parent['protocol'],bank=a.bank,parent_reference_cache_sha256=sha(a.reference_cache),query_boxes_per_mode=True,
        old_quality_replay_uses_original_boxes=True,maximal_permits_maximum_area=True,shipping_bank_unchanged=True)
    a.output.mkdir(parents=True,exist_ok=False);torch.save(output,a.output/'cache.pt')
    summary=dict(bank=a.bank,counts=counts,cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,
        original_reference_coordinates_and_roles_unchanged=True,model_api_calls=0,model_promoted=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'queries.json').write_text(json.dumps(queries,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
