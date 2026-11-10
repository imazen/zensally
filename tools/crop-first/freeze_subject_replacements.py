#!/usr/bin/env python3
"""Freeze reviewed replacement originals and their source-specific rejection record."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from sunburst_generate import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    root=a.output;verification=json.loads((a.run/'replacements-native-verification.json').read_text())
    compressed=json.loads((a.run/'replacements-jxl/summary.json').read_text())
    if verification['images']!=3 or verification['pending_attempts'] or compressed['images']!=3 or any(not r['variants']['lossless']['lossless_exact'] or not r['variants']['lossless']['independent_exact'] for r in compressed['records']):raise ValueError('complete3-source native/lossless verification required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['replacements','replacements-jxl','replacements-tasks','replacements-sol']:shutil.copytree(a.run/name,root/name)
    for name in ['authorization.json','replacement-plan.json','replacements-task-policy.json','replacements-native-verification.json','source-review-holds.json']:
        shutil.copy2(a.run/name,root/name)
    (root/'carry').mkdir();shutil.copy2(a.run/'expansion/budget.jsonl',root/'carry/budget.jsonl')
    for job in json.loads((a.run/'replacements/plan.json').read_text()):
        record=json.loads((a.run/'replacements'/(job['id']+'.json')).read_text());entry=a.run/'image-cache'/record['key'];shutil.copytree(entry,root/'image-cache'/entry.name)
    for path in (a.run/'replacements-sol').glob('sunburst-v4-*.json'):
        result=json.loads(path.read_text());entry=Path(result['entry']);shutil.copytree(entry,root/'vision-cache'/entry.name)
    names=['justfile','tools/crop-first/plan_subject_replacements.py','tools/crop-first/sunburst_generate.py','tools/crop-first/sunburst_costs.py',
        'tools/crop-first/sunburst_verify.py','tools/crop-first/sunburst_recompress.py','tools/crop-first/sunburst_dataset.py',
        'tools/crop-first/prepare_subject_tasks.py','tools/crop-first/freeze_subject_replacements.py','docs/experiments/SYNTHETIC_ART_DIRECTION.md']
    for name in names:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (root/'logs').mkdir()
    for prefix in ['replacements','replacements-jxl','replacements-sol']:
        for path in (Path.home()/'tmp').glob('zensally-subject-v4-'+prefix+'-20261109.*'):shutil.copy2(path,root/'logs'/path.name)
    manifest=dict(generation_sources=3,new_generation_usd=verification['new_published_rate_usd'],generation_total_usd=verification['total_published_rate_usd'],
        source_families_and_roles_inherited=True,old_sources_retained=True,rejected_sources_are_not_negative_crop_labels=True,
        full_size_coordinator_review=True,human_rated=False,face_coverage_certified=False,
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha((root/'SHA256SUMS.json').read_bytes()))))

if __name__=='__main__':main()

