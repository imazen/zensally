#!/usr/bin/env python3
"""Score fixed detector outputs against screened non-group person annotations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from analyze_efficientdet_family import iou,people
from artifact_identity import sha


def matches(row,boxes,width,height):
    targets=[b for b in boxes if not b['flags']['IsGroupOf']]
    converted=[dict(xywh=[b['box'][0],b['box'][1],b['box'][2]-b['box'][0],b['box'][3]-b['box'][1]]) for b in targets]
    predictions=[]
    for index,det in enumerate(people(row)):
        x,y,w,h=det['xywh'];score=max(c['score'] for c in det['categories'] if c['name']=='person')
        if not all(math.isfinite(v) for v in [x,y,w,h,score]) or w<=0 or h<=0:raise ValueError('invalid raw person prediction')
        predictions.append((score,index,dict(xywh=[x/width,y/height,w/width,h/height])))
    used=set()
    for _,_,prediction in sorted(predictions,key=lambda p:(-p[0],p[1])):
        choices=[(iou(prediction,target),-index,index) for index,target in enumerate(converted) if index not in used]
        if choices:
            overlap,_,index=max(choices)
            if overlap>=.5:used.add(index)
    return targets,used,len(predictions)


def aggregate(rows):
    target_count=sum(len(r['targets']) for r in rows);matched=sum(len(r['matched']) for r in rows)
    bins={}
    for kind in ['small','medium','large','occluded','not_occluded','truncated','not_truncated']:
        selected=[]
        for row in rows:
            for index,b in enumerate(row['targets']):
                area=(b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]);size='small' if area<.02 else 'medium' if area<.2 else 'large'
                belongs=(kind==size or kind==('occluded' if b['flags']['IsOccluded'] else 'not_occluded') or kind==('truncated' if b['flags']['IsTruncated'] else 'not_truncated'))
                if belongs:selected.append(index in row['matched'])
        bins[kind]=dict(targets=len(selected),matched=sum(selected),recall=sum(selected)/len(selected) if selected else None)
    return dict(sources=len(rows),sources_with_individual_targets=sum(bool(r['targets']) for r in rows),
        individual_targets=target_count,matched=matched,recall=matched/target_count if target_count else None,
        group_records=sum(r['group_records'] for r in rows),person_predictions=sum(r['predictions'] for r in rows),
        sources_above50_individual_targets=sum(len(r['targets'])>50 for r in rows),strata=bins)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','acquired','content','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    mask_path=a.content/'source-mask.json';freeze=json.loads((a.content/'mask-freeze.json').read_text())
    if sha(mask_path)!='e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80' or sha(mask_path)!=freeze['source_mask_sha256']:raise ValueError('independent frozen content mask differs')
    for name,digest in freeze['bindings'].items():
        if sha(a.content/name)!=digest:raise ValueError('review evidence changed after mask freeze')
    mask={r['id']:r for r in json.loads(mask_path.read_text())};manifest=json.loads((a.root/'manifest.json').read_text())
    sources={r['id']:r for r in json.loads((a.acquired/'manifest.json').read_text())['rows']}
    if len(mask)!=384 or set(mask)!=set(sources) or manifest['source_manifest_sha256']!=sha(a.acquired/'manifest.json'):raise ValueError('source/role/annotation binding differs')
    a.output.mkdir(parents=True,exist_ok=False)
    details=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_mask_sha256=sha(mask_path),original_protocol_sha256=sha(a.root/'scoring-protocol.json'),
        raw_prediction_boxes_normalized_without_clipping=True,matching='confidence-descending greedy unmatched IoU>=0.5',
        paired_bootstrap=dict(unit='source component, one image per component',resamples=5000,seed=31,percentiles=[2.5,97.5]),
        primary='screened development annotated non-group person recall',official_ap=False,face_recall=False,paid_model_api_calls=0)
    (a.output/'scoring-details.json').write_text(json.dumps(details,indent=2)+'\n')
    ids=[r['id'] for r in manifest['images']];all_rows={};models=[]
    for model in manifest['models']:
        path=a.root/'full'/model['id']/'results.json';results=json.loads(path.read_text())['rows']
        if [r['id'] for r in results]!=ids:raise ValueError('complete fixed384-source prediction order required')
        measured=[]
        for result in results:
            source=sources[result['id']];g=source['geometry'];m=mask[result['id']]
            targets,used,count=matches(result,source['boxes'],g['width'],g['height'])
            measured.append(dict(id=result['id'],role=m['role'],eligible=m['person_auxiliary_source_eligible'],
                targets=targets,matched=sorted(used),predictions=count,group_records=sum(b['flags']['IsGroupOf'] for b in source['boxes'])))
        all_rows[model['id']]=measured
        scores={}
        for eligible in [False,True]:
            for role in ['fit','dev','all']:
                subset=[r for r in measured if (not eligible or r['eligible']) and (role=='all' or r['role']==role)]
                scores[('screened' if eligible else 'all_sources')+'_'+role]=aggregate(subset)
        models.append(dict(id=model['id'],weight_bytes=model['bytes'],results_sha256=sha(path),scores=scores))
    primary_ids=[i for i in ids if mask[i]['role']=='dev' and mask[i]['person_auxiliary_source_eligible']]
    if len({mask[i]['family_id'] for i in primary_ids})!=len(primary_ids):raise ValueError('source bootstrap must match independent components')
    target_counts=np.array([len(next(r for r in all_rows[models[0]['id']] if r['id']==i)['targets']) for i in primary_ids],np.float64)
    samples=np.random.default_rng(31).integers(0,len(primary_ids),size=(5000,len(primary_ids)));denominator=target_counts[samples].sum(1)
    if not np.all(denominator>0):raise ValueError('bootstrap draw has no individual targets; preserve outputs')
    bootstrap={};point={}
    for model in models:
        lookup={r['id']:r for r in all_rows[model['id']]};hits=np.array([len(lookup[i]['matched']) for i in primary_ids],np.float64)
        bootstrap[model['id']]=hits[samples].sum(1)/denominator;point[model['id']]=float(hits.sum()/target_counts.sum())
    differences=[]
    for model in models:
        for baseline in ['int8-efficientdet_lite0','tf-efficientdet_lite0']:
            if model['id']==baseline:continue
            intervals=np.percentile(bootstrap[model['id']]-bootstrap[baseline],[2.5,97.5])
            differences.append(dict(model=model['id'],baseline=baseline,recall_difference=point[model['id']]-point[baseline],paired_source_ci95=intervals.tolist()))
    summary=dict(primary_sources=len(primary_ids),primary_individual_targets=int(target_counts.sum()),models=models,paired_development_differences=differences,
        source_mask_sha256=sha(mask_path),human_person_annotations_may_be_incomplete=True,face_recall_measured=False,
        precision_or_official_ap_measured=False,timing_measured=False,source_roles_changed=False,training_labels_admitted=False,paid_model_api_calls=0)
    for name,value in [('source-matches.json',all_rows),('summary.json',summary)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(primary_sources=len(primary_ids),primary_individual_targets=int(target_counts.sum()),models=[dict(id=m['id'],primary=m['scores']['screened_dev']) for m in models],paired_development_differences=differences),indent=2))


if __name__=='__main__':main()
