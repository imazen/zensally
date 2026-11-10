#!/usr/bin/env python3
"""Seal completed bank controls in place; preserve model states, witnesses and sources."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from artifacts import sha,verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('a sealed packet must not be rewritten')
    for bank in ['old-broad-maximal','hybrid']:
        root=a.root/bank;analysis=json.loads((root/'analysis/summary.json').read_text())
        if not analysis['all_selected_native_replays_exact'] or len(analysis['models'])!=12 or len(analysis['replay_controls'])!=3:raise ValueError('all15 verified fits required per bank')
        for job,count in [('fits',12),('replay-control-fits',3)]:
            status=json.loads((root/job/'PROGRESS.json').read_text())
            if status['stage']!='complete' or status['completed']!=count or status['pid'] is not None:raise ValueError('a fit job is still active or incomplete')
        browser=json.loads((root/'gallery/browser-miap-rank.json').read_text())
        if browser['checks']!=704 or browser['failures'] or browser['exceptions']:raise ValueError('complete browser verification required')
    names=['justfile','training/acceptable_crop_sets.py','training/test_acceptable_crop_sets.py',
        'training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','training/crop_fit.py',
        'tools/composition/fit_subject_sets.py','tools/composition/cache_subject_bank_control.py','tools/composition/subject_candidate_bank.py',
        'tools/composition/analyze_subject_sets.py','tools/composition/test_subject_analysis_baseline.py',
        'tools/composition/subject_sets_gallery.py','tools/composition/miap_rank_gallery.html','tools/composition/crop_clip.js',
        'tools/composition/miap_rank_browser_check.mjs','tools/composition/fit_unsplash_ranks.py','tools/composition/fit_extents.py',
        'tools/composition/artifact_identity.py','tools/composition/subject_policy.py','tools/crop-first/subject_geometry.py',
        'tools/crop-first/browser_check.py','tools/crop-first/artifacts.py','tools/composition/freeze_subject_bank_control.py',
        'docs/experiments/SUBJECT_BANK_CONTROL.md','benchmarks/subject_bank_controls_2026-11-09.md']
    for name in names:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (a.root/'logs').mkdir()
    for path in (Path.home()/'tmp').glob('zensally-subject-bank-*-20261109.log'):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(selected_fits=30,all_selected_native_replays_exact=True,model_promoted=False,api_calls=0,synthetic_image_gradients=False,
        test_inference=False,human_artistic_gold=False,complete_face_coverage_certified=False,reviewed_screenshots=4,retained_screenshots=20,
        parent_reference_packet='8818f1f01023e9552750e5965ee7e3374a42d571a944c7b9e913df1f470c1fa0',
        parent_rank_packet='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        native_parent_packet='21434d0f0ebbb88dc1ac0fa60b479af48e89d011cc03110a6e84427cc560a438',
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(path.relative_to(a.root)),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(a.root.rglob('*')) if path.is_file()]
    (a.root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(a.root,entries)
    print(json.dumps(dict(members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(a.root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
