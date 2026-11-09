#!/usr/bin/env python3
"""Freeze only the completed four-source pilot while later generation continues."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from sunburst_generate import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    source=a.run;root=a.output;root.mkdir(parents=True,exist_ok=False)
    for name in ['authorization.json','api-models.json','brand-search.json','native-verification.json','pilot-plan.json',
        'tote-native-count.json','tote-preview-count.json','birthday-native-count.json','birthday-preview-count.json',
        'tote-terra-result.json','person-speakers-white-terra-result.json','friends-camera-context-terra-result.json',
        'birthday-choice-candid-terra-result.json','birthday-sol-result.json','birthday-sol-task.json','birthday-owner-aliases.json']:
        shutil.copy2(source/name,root/name)
    for name in ['pilot','pilot-jxl','pilot-tasks','pilot-tasks-owned-rgb','birthday-repaired','geometry-audit','gallery','gallery-missing-resizer-attempt']:
        shutil.copytree(source/name,root/name)
    for result in root.glob('*result.json'):
        record=json.loads(result.read_text())
        entry=Path(record['entry']);shutil.copytree(entry,root/'vision-cache'/entry.name)
    for path in (source/'pilot').glob('*.json'):
        record=json.loads(path.read_text())
        if isinstance(record,dict) and 'key' in record:
            entry=source/'image-cache'/record['key'];shutil.copytree(entry,root/'image-cache'/entry.name)
    (root/'testing').mkdir()
    for name in ['budget-config.json','budget.jsonl']:
        shutil.copy2(source/'testing'/name,root/'testing'/('observation-'+name))
    shutil.copytree(source/'testing/token-counts',root/'testing/token-counts')
    producers=['justfile','tools/crop-first/sunburst_generate.py','tools/crop-first/sunburst_verify.py','tools/crop-first/sunburst_costs.py',
        'tools/crop-first/sunburst_recompress.py','tools/crop-first/sunburst_dataset.py','tools/crop-first/sunburst_subject_pilot.py',
        'tools/crop-first/api_budget.py','tools/crop-first/api_ledger.py','tools/crop-first/direct_zones.py','tools/crop-first/teacher.py',
        'tools/crop-first/photo_focus.py','tools/crop-first/subject_zones.py','tools/crop-first/prepare_subject_tasks.py',
        'tools/crop-first/repair_subject_aliases.py','tools/crop-first/subject_geometry.py','tools/crop-first/render_subject_geometry.py',
        'tools/crop-first/subject_browser_check.mjs','tools/crop-first/browser_check.py','tools/composition/subject_policy.py',
        'benchmarks/subject_choice_pilot_2026-11-09.md','docs/experiments/SUBJECT_SELECTION_V4.md',
        'tools/crop-first/freeze_subject_pilot.py']
    for name in producers:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    original=root/'source/original-generator-704fca9c.py'
    original.write_bytes(subprocess.check_output(['git','show','704fca9c:tools/crop-first/sunburst_generate.py']))
    if sha(original.read_bytes())!='707455e2b8a75664b33f2d14e2b75d7a5301d16b8bb8ea5fa8888832e4517225':raise ValueError('original generation source identity changed')
    (root/'logs').mkdir()
    for pattern in ['zensally-subject-v4-generation-20261109.*','zensally-subject-v4-prepare*-20261109.*','zensally-subject-v4-png-scope-20261109.*',
        'zensally-subject-v4-*-terra-20261109.*','zensally-subject-v4-birthday-sol-20261109.*','zensally-subject-v4-jxl-20261109.*',
        'zensally-subject-v4-browser-20261109.*']:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,root/'logs'/path.name)
    browser=json.loads((root/'gallery/browser-subject-choice.json').read_text())
    if browser['crops']!=38 or browser['rejections']!=1 or browser['exceptions']:raise ValueError('complete pilot browser checks required')
    manifest=dict(source_run=str(source),pilot_generation_cost_usd='0.125720',pilot_testing_cost_usd='0.09583425',
        pilot_cost_usd='0.22155425',generation_sources=4,terra_responses=4,sol_responses=1,
        excludes_later_expansion_and_architecture_costs=True,testing_journal_is_concurrent_observation_not_pilot_only=True,
        raw_invalid_terra_responses=1,explicit_alias_repairs=1,training_eligible=False,face_safety_certified=False,
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True))
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path.read_bytes())) for path in sorted(root.rglob('*')) if path.is_file()]
    with (root/'SHA256SUMS.json').open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha((root/'SHA256SUMS.json').read_bytes()))))

if __name__=='__main__':main()

