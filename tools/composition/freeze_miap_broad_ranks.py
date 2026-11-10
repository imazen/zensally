#!/usr/bin/env python3
"""Retain broader rank controls, regression witnesses and exact producer sources."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    if (root/'SHA256SUMS.json').exists():raise ValueError('broader packet already frozen')
    report=json.loads((root/'analysis/summary.json').read_text());browser=json.loads((root/'gallery/browser-miap-rank.json').read_text())
    if len(report['models'])!=12 or not report['all12_selected_native_replays_exact'] or not report['registered_new_photo_agreement_gate_passed']:
        raise ValueError('complete fixed broader comparison required')
    if browser['checks']!=704 or browser['exceptions'] or len(browser['screenshots'])!=10 or browser['screenshot_bytes']>96*1024**2:
        raise ValueError('complete bounded crop viewer checks required')
    for model in report['models']:
        path=root/'fits'/(model['arm']+'-seed'+str(model['seed']))/'best.pt'
        if sha(path)!=model['sha256']:raise ValueError('selected broader state differs')
    paths=['justfile','DATA_PROVENANCE.md','docs/experiments/MIAP_BROAD_RANKS.md','benchmarks/miap_broad_ranks_2026-11-09.md',
        'training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','training/crop_fit.py',
        'tools/composition/cache_miap_ranks.py','tools/composition/fit_miap_broad_ranks.py','tools/composition/analyze_miap_broad_ranks.py',
        'tools/composition/plan_miap_rank_review.py','tools/composition/miap_rank_gallery.py','tools/composition/miap_rank_gallery.html',
        'tools/composition/miap_rank_browser_check.mjs','tools/composition/crop_clip.js','tools/crop-first/browser_check.py',
        'tools/composition/analyze_unsplash_ranks.py','tools/composition/fit_unsplash_ranks.py','tools/composition/cached_focalnet.py',
        'tools/composition/constrained_bank.py','tools/composition/focalnet_bank.py','tools/composition/rank_targets.py',
        'tools/composition/fit_extents.py','tools/composition/artifact_identity.py','tools/composition/freeze_miap_broad_ranks.py']
    producers={}
    for name in paths:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        producers[name]=subprocess.check_output(['git','log','-1','--format=%H','--',name],text=True).strip()
    (root/'logs').mkdir()
    for prefix in ['rank-cache','broad-fits','broad-analysis','broad-gallery','broad-browser']:
        for path in Path('/home/lilith/tmp').glob('zensally-miap-'+prefix+'-20261109.*'):shutil.copy2(path,root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        producers=producers,new_cache_sha256=sha(root/'cache/cache.pt'),review_plan_sha256=sha(root/'review-plan.json'),
        auxiliary_parent_sha256='0da7cda46013a9cb0fc014f0d3bf4f024eb0a05b851c1115bb6f1b5d29cbe7c8',
        old_rank_parent_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        new_photo_agreement_gate_passed=True,old_cohort_regression_retained=True,model_promoted=False,
        face_coverage_certified=False,human_artistic_quality_measured=False,new_test_inference=False,paid_model_api_calls=0)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    total=sum(r['bytes'] for r in entries)
    if total>3*1024**3:raise ValueError('cache/fit artifact allowance exceeded')
    with (root/'SHA256SUMS.json').open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=total,manifest_sha256=sha(root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
