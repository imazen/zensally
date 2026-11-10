#!/usr/bin/env python3
"""Snapshot the completed pilot without freezing its future spending ledger."""
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from sunburst_generate import journal
from direct_zones import sha
from artifacts import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    summary=json.loads((a.root/'analysis-phase1/summary.json').read_text());states=journal(a.root/'testing/budget.jsonl')
    if summary['native_responses']!=56 or summary['proposal_sources_v2']!=12 or summary['ready_for_rendered_review_v2']!=14 or summary['invalid_annotations'] or summary['pending_reservations'] or Decimal(summary['ledger_native_difference_usd'])!=0:raise ValueError('complete first-pilot native reconciliation required')
    if len(states)!=56 or any(row['state']!='settled' for row in states.values()) or sum(Decimal(r['usd']) for r in states.values())!=Decimal(summary['settled_ledger_usd']):raise ValueError('pilot ledger changed after analysis')
    for name,count in [('corridor-v2-results',12),('review-results-v2',42)]:
        status=json.loads((a.root/name/'PROGRESS.json').read_text())
        if status['pid'] is not None or status['stage']!='complete' or status['completed']!=count or status['valid']!=count or status['errors']:raise ValueError('pilot worker not complete')
    browser=json.loads((a.root/'gallery-phase1/browser-corridor.json').read_text())
    if browser['groups']!=14 or browser['decoded']!=98 or browser['geometry']!=70 or browser['failures'] or browser['exceptions'] or len(browser['screenshots'])!=4:raise ValueError('complete bounded browser evidence required')
    cache_keys=set()
    for entry in (a.root/'vision-cache').iterdir():
        if entry.is_dir() and re.fullmatch('[0-9a-f]{64}',entry.name):
            cache_keys.add(entry.name)
            for name,digest in json.loads((entry/'hashes.json').read_text()).items():
                if sha((entry/name).read_bytes())!=digest:raise ValueError('native cache differs')
    if cache_keys!=set(states):raise ValueError('native response/ledger membership differs')
    def copy(source,target):
        path=Path(source);relative=path.relative_to(a.root)
        immutable_media=path.suffix in ['.png','.rgb','.rgb8']
        immutable_response=len(relative.parts)>=3 and relative.parts[0]=='vision-cache' and bool(re.fullmatch('[0-9a-f]{64}',relative.parts[1]))
        if immutable_media or immutable_response:os.link(source,target)
        else:shutil.copy2(source,target)
        return target
    shutil.copytree(a.root,a.output,copy_function=copy,ignore=shutil.ignore_patterns('*.lock'))
    names=['justfile','tools/crop-first/api_budget.py','tools/crop-first/api_ledger.py','tools/crop-first/sunburst_generate.py',
        'tools/crop-first/direct_zones.py','tools/crop-first/teacher.py','tools/crop-first/photo_focus.py','tools/crop-first/subject_zones.py',
        'tools/crop-first/subject_geometry.py','tools/crop-first/crop_corridors.py','tools/crop-first/test_crop_corridors.py',
        'tools/crop-first/propose_crop_corridors.py','tools/crop-first/test_corridor_proposals.py',
        'tools/crop-first/label_quality.py','tools/crop-first/test_label_quality.py','tools/crop-first/test_label_pilot_budget.py',
        'tools/crop-first/prepare_label_quality.py','tools/crop-first/rendered_pixels.py','tools/crop-first/render_crop_corridors.py',
        'tools/crop-first/prepare_corridor_review_batch.py','tools/crop-first/batch_subject_zones.py',
        'tools/crop-first/analyze_corridor_reviews.py','tools/crop-first/corridor_review_gallery.py',
        'tools/crop-first/corridor_browser_check.mjs','tools/crop-first/browser_check.py',
        'tools/crop-first/record_label_progress.py','tools/crop-first/artifacts.py','tools/crop-first/freeze_label_pilot.py',
        'tools/composition/subject_policy.py','benchmarks/crop_label_quality_2026-11-09.md',
        'benchmarks/real_attention_inputs_2026-11-09.md','docs/experiments/LABEL_QUALITY_AUDIT.md']
    for name in names:
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    directory=a.output/'source/as-run';directory.mkdir()
    for revision,name,label in [('14ef3a65','tools/crop-first/propose_crop_corridors.py','initial-v1-proposer'),
        ('960c8817','tools/crop-first/propose_crop_corridors.py','v2-proposer'),
        ('c7cdead6','tools/crop-first/render_crop_corridors.py','first-renderer')]:
        (directory/(label+'.py')).write_bytes(subprocess.check_output(['git','show',revision+':'+name]))
    sources=a.output/'original-sources';sources.mkdir()
    for row in json.loads((a.root/'plan.json').read_text())['cases']:
        path=Path(row['image']['path']);raw=path.read_bytes()
        if sha(raw)!=row['image']['sha256']:raise ValueError('source preview differs')
        (sources/(row['case_id']+'.png')).write_bytes(raw)
    logs=a.output/'logs';logs.mkdir()
    for pattern in ['zensally-corridor-*-20261109.log','zensally-label-quality-*-20261109.log']:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,logs/path.name)
    manifest=dict(stage='completed first label-quality pilot; no excellent intervals admitted',native_responses=56,
        settled_usd=summary['settled_ledger_usd'],new_authorization_usd='20',pilot_cap_usd='2',pending_reservations=0,
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        hardlink_policy='Local snapshot hardlinks immutable media and published content-addressed responses; mutable metadata/ledger copied. Independent NAS readback verifies every payload.',
        budget_ledger_is_an_immutable_prefix_here=True,live_source_root=str(a.root),human_artistic_gold=False,
        face_coverage_certified=False,model_promoted=False,training_started=False,test_inference=False,
        browser=dict(decoded=98,geometry=70,screenshots=4,visually_inspected_screenshots=4),
        real_attention_parent='86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3')
    (a.output/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.output)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(a.output.rglob('*')) if p.is_file()]
    (a.output/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(a.output,entries)
    print(json.dumps(dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha((a.output/'SHA256SUMS.json').read_bytes()))))


if __name__=='__main__':main()
