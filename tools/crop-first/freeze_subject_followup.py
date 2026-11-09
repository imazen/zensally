#!/usr/bin/env python3
"""Freeze completed Sol controls, cross-teacher audit and bank geometry evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from sunburst_generate import journal, exposure
from artifacts import sha, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','bank','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    verification=json.loads((a.run/'responses-212-native-verification.json').read_text())
    analysis=json.loads((a.run/'sol-analysis/summary.json').read_text())
    cross=json.loads((a.run/'cross-teacher/summary.json').read_text())
    keys={r['cache_key'] for r in verification['records']}
    states=journal(a.run/'testing/budget.jsonl')
    if verification['responses']!=212 or verification['pending_attempts'] or set(states)!=keys or any(r['state']!='settled' for r in states.values()):
        raise ValueError('complete settled212-response snapshot required')
    if str(exposure(states))!=verification['settled_native_total_usd']:raise ValueError('ledger verification differs')
    if len(analysis['models'])!=12 or len(analysis['replay_controls'])!=3 or not analysis['all_selected_native_replays_exact']:
        raise ValueError('complete15-fit native replay verification required')
    if not cross['input_panels_bit_identical'] or cross['paid_model_api_calls'] or cross['model_promoted']:raise ValueError('unexpected cross-teacher protocol')
    for folder,count in [('sol-fits',12),('sol-replay-control-fits',3)]:
        progress=json.loads((a.run/folder/'PROGRESS.json').read_text())
        if progress['stage']!='complete' or progress['completed']!=count:raise ValueError('incomplete fit job')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['sol-primary-tasks','sol-primary','sol-real72','sol-set-cache','sol-fits','sol-replay-control-fits','sol-analysis','cross-teacher']:
        shutil.copytree(a.run/name,a.output/name)
    shutil.copytree(a.bank,a.output/'bank-density')
    for name in ['authorization.json','responses-212-native-verification.json','dataset-91-remote-verification.json','source-review-holds.json']:
        shutil.copy2(a.run/name,a.output/name)
    for key in keys:shutil.copytree(a.run/'vision-cache'/key,a.output/'vision-cache'/key)
    (a.output/'testing').mkdir()
    for name in ['budget.jsonl','budget-config.json','rates-extension-v1.json']:
        shutil.copy2(a.run/'testing'/name,a.output/'testing'/name)
    shutil.copytree(a.run/'testing/token-counts',a.output/'testing/token-counts')
    (a.output/'generation').mkdir();shutil.copy2(a.run/'replacements/budget.jsonl',a.output/'generation/budget.jsonl')
    names=['justfile','training/acceptable_crop_sets.py','training/local_rank_distillation.py','training/crop_regions.py',
        'training/crop_ranker.py','training/model_v3ds.py','training/crop_fit.py',
        'tools/composition/fit_subject_sets.py','tools/composition/analyze_subject_sets.py',
        'tools/composition/audit_subject_cross_teacher.py','tools/composition/audit_subject_bank_density.py',
        'tools/composition/subject_candidate_bank.py','tools/composition/test_subject_candidate_bank.py',
        'tools/composition/subject_policy.py','tools/composition/artifact_identity.py',
        'tools/composition/fit_unsplash_ranks.py','tools/composition/fit_extents.py',
        'tools/crop-first/cache_subject_sets.py','tools/crop-first/subject_geometry.py',
        'tools/crop-first/subject_zones.py','tools/crop-first/direct_zones.py','tools/crop-first/teacher.py',
        'tools/crop-first/photo_focus.py','tools/crop-first/api_budget.py','tools/crop-first/api_ledger.py',
        'tools/crop-first/batch_subject_zones.py','tools/crop-first/derive_subject_response_view.py',
        'tools/crop-first/verify_subject_responses.py','tools/crop-first/verify_dataset_blobs.py',
        'tools/crop-first/sunburst_generate.py','tools/crop-first/sunburst_costs.py',
        'tools/crop-first/freeze_subject_followup.py','tools/crop-first/artifacts.py',
        'docs/experiments/SYNTHETIC_ART_DIRECTION.md','benchmarks/subject_choice_followup_2026-11-09.md']
    for name in names:
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (a.output/'source/original-density-scout.py').write_bytes(subprocess.check_output(['git','show','0007d4ca:tools/composition/audit_subject_bank_density.py']))
    (a.output/'logs').mkdir()
    for pattern in ['zensally-subject-sol-*-20261109.*','zensally-subject-cross-teacher-20261109.*','zensally-subject-v4-bank-density*-20261109.*']:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,a.output/'logs'/path.name)
    manifest=dict(responses=212,selected_sol_fits=15,all_selected_native_replays_exact=True,
        testing_snapshot_usd=verification['settled_native_total_usd'],generation_snapshot_usd='1.141760',total_snapshot_usd='6.42712175',
        model_promoted=False,test_inference=False,face_coverage_certified=False,human_artistic_gold=False,
        parent_packets=dict(first_fits='8818f1f01023e9552750e5965ee7e3374a42d571a944c7b9e913df1f470c1fa0',
            replacements='65c6f1869b0b50ed7ec5d97334192dff039e6f0450f9ead9299edf731aa7cd91',
            broader_rank='21434d0f0ebbb88dc1ac0fa60b479af48e89d011cc03110a6e84427cc560a438'),
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    (a.output/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(path.relative_to(a.output)),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(a.output.rglob('*')) if path.is_file()]
    if sum(r['bytes'] for r in entries)>2*1024**3:raise ValueError('2GiB followup archive allowance exceeded')
    (a.output/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n')
    verify(a.output,entries)
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(a.output/'SHA256SUMS.json'))))


if __name__=='__main__':main()
