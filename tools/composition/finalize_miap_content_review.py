#!/usr/bin/env python3
"""Freeze all visual observations and independent source masks before scoring."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    data=json.loads((a.root/'sources.json').read_text());browser=json.loads((a.root/'browser-miap-content.json').read_text())
    if len(data['sources'])!=384 or browser['decoded']!=384 or browser['region_boxes']!=2518 or browser['exceptions'] or len(browser['sheets'])!=16 or browser['screenshot_bytes']>96*1024**2:raise ValueError('complete source/box/browser evidence required')
    with (a.root/'visual-tags.psv').open() as stream:tags=list(csv.DictReader(stream,delimiter='|'))
    if len(tags)!=384 or {int(r['index']) for r in tags}!=set(range(384)):raise ValueError('all384 source observations required')
    by_index={int(r['index']):dict(r) for r in tags}
    with (a.root/'visual-tags-corrections.psv').open() as stream:corrections=list(csv.DictReader(stream,delimiter='|'))
    for correction in corrections:
        row=by_index[int(correction['index'])]
        if (row['scope'],row['targets'])!=(correction['old_scope'],correction['old_targets']):raise ValueError('correction does not bind prior observation')
        row.update(scope=correction['scope'],targets=correction['targets'],note=row['note']+' Full-preview correction: '+correction['note'])
    observations=[];mask=[]
    for index in range(384):
        tag=by_index[index];source=data['sources'][index]
        if tag['scope'] not in {'photo','document_art','uncertain'} or tag['targets'] not in {'person_candidate','ambiguous_depiction','unknown'} or not tag['note']:raise ValueError('invalid source observation')
        if source['index']!=index or source['role'] not in {'fit','dev'} or sha(a.root/source['preview'])!=source['preview_sha256']:raise ValueError('observed source/preview binding differs')
        photo=tag['scope']=='photo';person=tag['targets']=='person_candidate';relationship=source['relationship_flagged']
        observations.append(dict(tag,id=source['id'],role=source['role'],family_id=source['family_id'],
            preview_sha256=source['preview_sha256'],evidence_sheet='screenshots/sheet-'+str(index//24).zfill(2)+'.png',
            reviewer='Codex source-preview inspection in the active conversation',human_rated=False,
            face_coverage_certified=False,artistic_quality_rated=False))
        mask.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],photo_candidate=photo,
            person_target_candidate=person,relationship_flagged=relationship,
            composition_source_eligible=photo and not relationship,person_auxiliary_source_eligible=photo and person and not relationship,
            original_canonical_roles_changed=False,training_labels_admitted=False,face_coverage_certified=False))
    coverage={}
    for role in ['fit','dev','all']:
        subset=[r for r in mask if role=='all' or r['role']==role];obs=[r for r in observations if role=='all' or r['role']==role]
        coverage[role]=dict(sources=len(subset),scope=dict(Counter(r['scope'] for r in obs)),targets=dict(Counter(r['targets'] for r in obs)),
            relationship_flagged=sum(r['relationship_flagged'] for r in subset),
            composition_eligible=sum(r['composition_source_eligible'] for r in subset),person_auxiliary_eligible=sum(r['person_auxiliary_source_eligible'] for r in subset))
    record=dict(observations=observations,coverage=coverage,raw_source_observations=384,full_preview_corrections=len(corrections),
        source_roles_changed=False,training_labels_admitted=False,face_coverage_certified=False,artistic_quality_rated=False,
        caption_hints_used_for_review_order_only=True,caption_absence_is_not_acceptance=True,caption_invalid_sources_manually_reviewed=True,
        detector_predictions_used_for_source_mask=False,new_test_inference=False,paid_model_api_calls=0)
    for name,value in [('content-review.json',record),('source-mask.json',mask)]:
        with (a.root/name).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    binding=dict(checked_utc=datetime.now(timezone.utc).isoformat(),source_mask_sha256=sha(a.root/'source-mask.json'),
        bindings={name:sha(a.root/name) for name in ['sources.json','visual-tags.psv','visual-tags-corrections.psv','browser-miap-content.json','content-review.json']},
        mask_frozen_before_detector_quality_scoring=True,coverage=coverage,training_labels_admitted=False)
    with (a.root/'mask-freeze.json').open('x') as stream:stream.write(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(binding,indent=2))


if __name__=='__main__':main()
