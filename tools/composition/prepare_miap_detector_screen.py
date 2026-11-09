#!/usr/bin/env python3
"""Bind fixed detector checkpoints before independently reviewed person recall."""
import argparse
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['acquired','family','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.family/'SHA256SUMS.json')!='43b0cc63ef8bd17afdc48884cf029ddf791462c18ae87bbd2f09a0e44d9d836d':raise ValueError('fixed five-checkpoint packet differs')
    original=json.loads((a.family/'manifest.json').read_text());models=[]
    for model in original['models']:
        path=a.family/'models'/Path(model['path']).name
        if sha(path)!=model['sha256']:raise ValueError('detector weight differs')
        models.append(dict(model,path=str(path)))
    source=json.loads((a.acquired/'manifest.json').read_text());rows=source['rows']
    if len(rows)!=384 or source['protocol']['plan_packet_sha256']!='e5baf11ae1673af8b1feb98e790cbd17f93976f5c37ebe3888af3d508bd0a5db':raise ValueError('fixed acquired cohort differs')
    images=[]
    for row in rows:
        path=a.acquired/row['id'].split(':')[1]/'preview.rgb';g=row['geometry']
        if row['source']['role'] not in {'fit','dev'} or row['training_eligible'] or sha(path)!=row['preview.rgb_sha256']:raise ValueError('source role/pixel contract differs')
        images.append(dict(id=row['id'],width=g['width'],height=g['height'],rgb=str(path),rgb_sha256=row['preview.rgb_sha256'],benchmark=False,role=row['source']['role']))
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        images=images,models=models,source_manifest_sha256=sha(a.acquired/'manifest.json'),weight_packet_sha256=sha(a.family/'SHA256SUMS.json'),
        confidence_threshold=.3,max_results=50,preprocessing='MediaPipe Task CPU on identical owned RGB',
        training_eligible=False,face_coverage_certified=False,timing_measured=False,paid_model_api_calls=0)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol=dict(primary='annotated non-group person recall atIoU0.5 on independently content-screened development sources',
        all_sources_secondary=True,confidence_threshold=.3,max_results=50,match='descending confidence, best unmatched IoU>=0.5',
        group_boxes='not individual recall targets; report separately, no precision/AP claim',
        primary_source_mask='photo and candidate human targets, no relationship flag, independent of detector predictions',
        size_strata='normalized individual box area:small<0.02, medium<0.2, large>=0.2',
        extra_strata=['source role','occlusion','truncation'],inference_all_five_checkpoints_fixed_before_scoring=True,
        do_not_read_detector_quality_before_content_review=True,canonical_holdout_pixels_opened=False,new_test_inference=False,
        human_annotations_may_be_incomplete=True,face_recall_measured=False,model_promoted=False,paid_model_api_calls=0)
    (a.output/'scoring-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(sources=len(images),models=len(models),expected_inference_calls=len(images)*len(models))))


if __name__=='__main__':main()
