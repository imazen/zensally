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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['transfer','direct','cues']);p.add_argument('root',type=Path);a=p.parse_args()
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
    elif a.kind=='direct':
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
    else:
        local=json.loads((a.root/'local/summary.json').read_text());manifest=json.loads((a.root/'manifest.json').read_text())
        guard=json.loads((a.root/'guard-audit/summary.json').read_text())
        if len(local)!=4 or any(r['returncode'] for r in local) or len(manifest['images'])!=72 or len(result['local_scopes'])!=4 or guard['model_promoted']:raise ValueError('complete72-source/four-cue evaluation and guard audit required')
        names+=['tools/composition/prepare_subject_face_cues.py','tools/composition/audit_subject_face_cues.py',
            'tools/composition/audit_local_crop_guards.py','tools/composition/pose_head_bounds.py','tools/composition/subject_policy.py',
            'tools/crop-first/subject_geometry.py','tools/runtime-eval/face_audit.py','tools/runtime-eval/run_models.py',
            'tools/runtime-eval/mediapipe_probe.py','benchmarks/subject_face_cues_2026-11-09.md']
        patterns=['zensally-subject-face-cues-*-20261109.log'];initial=None
    names+=['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py',
        'training/crop_fit.py','tools/composition/fit_unsplash_ranks.py','tools/composition/fit_extents.py']
    for name in names:
        target=a.root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    if initial:(a.root/'source/initial-failed-producer.py').write_bytes(subprocess.check_output(['git','show',initial[0]+':'+initial[1]]))
    (a.root/'logs').mkdir()
    for pattern in patterns:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(kind=a.kind,paid_api_calls=0,model_promoted=False,test_inference=False,source_roles_changed=False,
        human_artistic_gold=False,face_coverage_certified=False,
        parent_packets=dict(subject_followup='90b7c63a38b247bad5cd8a55bf15c7b8b05e04aeefc348dbf388fc0c7c2000d3',
            bank_controls='fb73334275761c51483b0fb77c3a63e80c80bb290404dcb403ea8483b859beb6',
            original_rank='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e'),
        source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip())
    if a.kind=='cues':manifest['parent_packets'].update(owned_acquisition='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1',
        external_model_inventory='4a58cd6aa5aa966bbaba3cd02506d7cdba545bde9371974421242c0effd4689b',
        sol_query_bank='adb91b9cf3bf44515389ec12d7caa3de13536a48a78c454d5c0619759058d419')
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    (a.root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(a.root,entries)
    print(json.dumps(dict(kind=a.kind,members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(a.root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
