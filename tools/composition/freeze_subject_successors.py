#!/usr/bin/env python3
"""Seal completed cached bank-transfer or direct-crop suitability evidence in place."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from artifacts import sha,verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['transfer','direct']);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('sealed packet must not be rewritten')
    result=json.loads((a.root/'analysis/summary.json').read_text())
    if result.get('api_calls',result.get('paid_api_calls')) or result['model_promoted']:raise ValueError('unexpected spending or promotion')
    names=['justfile','tools/composition/artifact_identity.py','tools/crop-first/artifacts.py','tools/composition/freeze_subject_successors.py']
    if a.kind=='transfer':
        if not result['cross_teacher_model_inputs_bit_identical'] or result['checkpoint_selection_changed']:raise ValueError('fixed-head transfer conditions differ')
        names+=['tools/composition/audit_subject_bank_transfer.py','tools/composition/cache_subject_bank_control.py',
            'tools/composition/subject_candidate_bank.py','tools/composition/subject_policy.py','tools/crop-first/subject_geometry.py',
            'benchmarks/subject_bank_transfer_2026-11-09.md']
        patterns=['zensally-subject-bank-transfer-*-20261109.log'];initial=('2133e9d9','tools/composition/audit_subject_bank_transfer.py')
    else:
        status=json.loads((a.root/'fits/PROGRESS.json').read_text())
        if status['stage']!='complete' or status['completed']!=9 or status['pid'] is not None:raise ValueError('nine complete fits required')
        rows=json.loads((a.root/'fits/summary.json').read_text())
        if len(rows)!=9 or any(not r['native_replay_exact'] for r in rows) or len(result['cpu_gpu_native_checks'])!=9 or not result['progress_repair_initial_outputs_exact']:raise ValueError('complete direct native verification required')
        for row in rows:
            if sha(a.root/'fits'/(row['arm']+'-seed'+str(row['seed']))/'best.pt')!=row['weights_sha256']:raise ValueError('selected checkpoint differs')
        names+=['training/direct_crop.py','training/test_direct_crop.py','tools/composition/cache_direct_crop.py',
            'tools/composition/fit_direct_crop.py','tools/composition/analyze_direct_crop.py','docs/experiments/DIRECT_CROP_SCOUT.md',
            'benchmarks/direct_crop_scout_2026-11-09.md']
        patterns=['zensally-direct-crop-*-20261109.log'];initial=('e23f3124','tools/composition/fit_direct_crop.py')
    names+=['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py',
        'training/crop_fit.py','tools/composition/fit_unsplash_ranks.py','tools/composition/fit_extents.py']
    for name in names:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (a.root/'source/initial-failed-producer.py').write_bytes(subprocess.check_output(['git','show',initial[0]+':'+initial[1]]))
    (a.root/'logs').mkdir()
    for pattern in patterns:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(kind=a.kind,paid_api_calls=0,model_promoted=False,test_inference=False,source_roles_changed=False,
        human_artistic_gold=False,face_coverage_certified=False,
        parent_packets=dict(subject_followup='90b7c63a38b247bad5cd8a55bf15c7b8b05e04aeefc348dbf388fc0c7c2000d3',
            bank_controls='fb73334275761c51483b0fb77c3a63e80c80bb290404dcb403ea8483b859beb6',
            original_rank='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e'),
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    (a.root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(a.root,entries)
    print(json.dumps(dict(kind=a.kind,members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(a.root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
