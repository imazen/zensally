#!/usr/bin/env python3
"""Freeze the completed114-response/15-fit study while Sol-primary work continues."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from sunburst_generate import sha,journal,exposure

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    root=a.output;verification=json.loads((a.run/'responses-114-native-verification.json').read_text())
    analysis=json.loads((a.run/'analysis/summary.json').read_text())
    if verification['responses']!=114 or verification['pending_attempts'] or len(analysis['models'])!=12 or len(analysis['replay_controls'])!=3 or not analysis['all_selected_native_replays_exact']:raise ValueError('complete fixed114response/15fit study required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['set-cache','fits','replay-control-fits','analysis','fit-gallery','real-tasks','real-terra','real-sol-tasks','real-sol','expansion-tasks','expansion-terra']:
        shutil.copytree(a.run/name,root/name)
    for name in ['authorization.json','api-models.json','brand-search.json','responses-114-native-verification.json','responses-78-native-verification.json',
        'real-sol-comparison.json','real-sol-comparison.initial.bak.json','dataset-88-remote-verification.json',
        'astra-consult-result.json','expansion-task-policy.json','expansion-native-verification.json','native-verification.json']:
        shutil.copy2(a.run/name,root/name)
    keys={r['cache_key'] for r in verification['records']}
    for key in keys:shutil.copytree(a.run/'vision-cache'/key,root/'vision-cache'/key)
    (root/'testing').mkdir()
    for name in ['budget-config.json','rates-extension-v1.json']:shutil.copy2(a.run/'testing'/name,root/'testing'/name)
    prefix=[]
    for line in (a.run/'testing/budget.jsonl').read_bytes().splitlines(keepends=True):
        if json.loads(line)['key'] not in keys:break
        prefix.append(line)
    (root/'testing/budget.jsonl').write_bytes(b''.join(prefix));states=journal(root/'testing/budget.jsonl')
    if set(states)!=keys or any(r['state']!='settled' for r in states.values()) or str(exposure(states))!=verification['settled_native_total_usd']:raise ValueError('exact ledger prefix does not reproduce the114-response snapshot')
    shutil.copytree(a.run/'testing/token-counts',root/'testing/token-counts-observation')
    names=['justfile','training/acceptable_crop_sets.py','training/test_acceptable_crop_sets.py','training/local_rank_distillation.py',
        'training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','training/crop_fit.py',
        'tools/composition/fit_subject_sets.py','tools/composition/analyze_subject_sets.py','tools/composition/subject_sets_gallery.py',
        'tools/composition/miap_rank_gallery.html','tools/composition/miap_rank_browser_check.mjs','tools/composition/crop_clip.js',
        'tools/composition/fit_unsplash_ranks.py','tools/composition/fit_extents.py','tools/composition/artifact_identity.py',
        'tools/crop-first/api_budget.py','tools/crop-first/api_ledger.py','tools/crop-first/direct_zones.py','tools/crop-first/subject_zones.py',
        'tools/crop-first/teacher.py','tools/crop-first/photo_focus.py','tools/crop-first/subject_geometry.py','tools/crop-first/prepare_subject_tasks.py',
        'tools/crop-first/plan_real_subject_tasks.py','tools/crop-first/batch_subject_zones.py','tools/crop-first/verify_subject_responses.py',
        'tools/crop-first/compare_subject_teachers.py','tools/crop-first/cache_subject_sets.py','tools/crop-first/consult_crop_architecture.py',
        'tools/crop-first/browser_check.py','tools/crop-first/sunburst_generate.py','tools/crop-first/sunburst_costs.py',
        'tools/composition/subject_policy.py','benchmarks/subject_choice_fitting_2026-11-09.md',
        'docs/experiments/SUBJECT_SELECTION_V4.md','tools/crop-first/freeze_subject_fits.py']
    for name in names:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    originals=[('d32f4bfa','tools/composition/fit_subject_sets.py','main12-fitter.py'),
        ('4ef60926','tools/composition/fit_subject_sets.py','replay3-fitter.py'),
        ('24f38c38','tools/crop-first/cache_subject_sets.py','original-reference-cache.py'),
        ('e6b4441d','tools/crop-first/api_budget.py','original-five-response-budget.py')]
    for revision,path,name in originals:(root/'source'/name).write_bytes(subprocess.check_output(['git','show',revision+':'+path]))
    (root/'logs').mkdir()
    for prefix in ['fits','replay-control','bank-cache','fit-gallery','fit-browser','real-terra','real-sol','expansion-terra','astra']:
        for path in (Path.home()/'tmp').glob('zensally-subject-v4-'+prefix+'-20261109.*'):shutil.copy2(path,root/'logs'/path.name)
    manifest=dict(responses=114,selected_fits=15,all_selected_native_replays_exact=True,testing_snapshot_usd=verification['settled_native_total_usd'],
        generation_snapshot_usd='1.022685',total_snapshot_usd='2.91223375',excludes_later_sol_primary=True,
        token_count_directory_is_a_concurrent_observation=True,face_coverage_certified=False,model_promoted=False,test_inference=False,
        parent_packets=dict(pilot='5c4494e41afee6f79c3ca7394576b7e8e27978fdf72ceca2af6ea657aa7f4947',
            generated_expansion='98aa1cc81d13ee7aa1aa338bd6fd53a263e2dcd22d933f6a3b565d851f28ed7c',
            old_rank='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
            broader_rank='21434d0f0ebbb88dc1ac0fa60b479af48e89d011cc03110a6e84427cc560a438'),
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file()]
    if sum(r['bytes'] for r in entries)>3*1024**3:raise ValueError('3GiB study archive allowance exceeded')
    (root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha((root/'SHA256SUMS.json').read_bytes()))))

if __name__=='__main__':main()

