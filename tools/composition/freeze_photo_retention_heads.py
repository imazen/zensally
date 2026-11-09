#!/usr/bin/env python3
"""Preserve all retention controls and bind selections before supplementary scoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--natural-loss',action='store_true');a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('packet already frozen')
    analysis=json.loads((a.root/'analysis.json').read_text())
    progress=json.loads((a.root/'fits/progress.json').read_text())
    runs=json.loads((a.root/'fits/summary.json').read_text())
    if (analysis['exact_selected_replays']!=9 or analysis['development_families']!=14
        or progress['pid'] is not None or progress['completed_runs']!=9 or len(runs)!=9):
        raise ValueError('all nine completed, replay-qualified controls required')
    if sha(a.root/'features.pt')!='2b36c1215840dbc889481f6716eaddb5ba43d4b5af6fa434e3d1be6e91a9a3f1':
        raise ValueError('qualified feature cache differs')
    bindings=[]
    for run in runs:
        path=a.root/'fits'/(run['arm']+'-seed'+str(run['seed']))/'best.pt'
        if sha(path)!=run['best_weights_sha256']:raise ValueError('selected state differs')
        bindings.append(dict(arm=run['arm'],seed=run['seed'],selected_epoch=run['best_epoch'],
            path=str(path.resolve()),sha256=sha(path)))
    producer=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip()
    (a.root/'supplementary-selection-plan.json').write_text(json.dumps(dict(build_commit=producer,
        fixed_models=bindings,cohort_sha256='a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70',
        cached_reference_packet_sha256='9fe44aa3645d379ab1ab925731f4710674f84b03171908026b1b3210a553e9f4',
        source_role='supplementary_dev',sources=15,model_selection_changes_allowed=False,
        supplementary_retention_scored=False,new_test_inference=False,paid_model_api_calls=0),indent=2)+'\n')
    paths=[Path('justfile'),Path('docs/experiments/PHOTO_RETENTION_HEADS.md'),
        Path('benchmarks/photo_retention_heads_2026-11-09.md')]
    if a.natural_loss:
        protocol=json.loads((a.root/'fits/protocol.json').read_text())
        if protocol.get('class_balance')!='natural candidate frequency within each source/query':raise ValueError('registered natural-loss controls required')
        paths += [Path('docs/experiments/RETENTION_LOSS_CONTROL.md'),Path('benchmarks/retention_natural_loss_2026-11-09.md')]
        paths += [Path('tools/composition')/n for n in ['prepare_retention_loss_control.py','test_retention_loss.py','compare_retention_losses.py']]
    paths+=[Path('tools/composition')/n for n in ['artifact_identity.py','cache_photo_retention.py',
        'fit_photo_retention.py','audit_photo_retention_heads.py','freeze_photo_retention_heads.py',
        'fit_extents.py','analyze_unsplash_ranks.py']]
    paths+=[Path('training')/n for n in ['local_rank_distillation.py','crop_fit.py','crop_ranker.py','crop_regions.py','model_v3ds.py']]
    producers={}
    for path in paths:
        target=a.root/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        producers[str(path)]=subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (a.root/'logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-photo-retention*20261109.log'):
        shutil.copy2(path,a.root/'logs'/path.name)
    if a.natural_loss:
        for pattern in ['zensally-retention-natural*20261109.log','zensally-retention-loss-unit*20261109.log']:
            for path in Path('/home/lilith/tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    else:
        shutil.copy2('/home/lilith/tmp/zensally-photo-retention-cache-20261109.time',a.root/'cache.time')
    manifest=dict(build_commit=producer,source_producers=producers,completed_fits=9,selected_replays_exact=9,
        fitted_sources=72,fitted_families=61,development_sources=15,development_families=14,
        reference_packet_sha256='a16e2bb9f324fe1ea04ef5392e538d61c9cd62ffeabd534a80cdc13e024905c8',
        rank_packet_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        training_labels='fitting-only noisy known-box containment/fraction; no complete subject or human artistic labels',
        original_encoder_updated=False,original_composition_head_updated=False,source_roles_changed=False,
        face_coverage_certified=False,new_test_inference=False,paid_model_api_calls=0,
        model_promoted=False,diagnostic_maps_saved=False)
    if a.natural_loss:
        manifest.update(ablation='natural candidate-frequency BCE; offer threshold unchanged',
            baseline_packet_sha256='3e603f65c94814b360fe8bee64484cffe07c39024ef32a4dda6e062c85a1698a',
            feature_cache_recomputed=False)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(a.root.rglob('*')) if p.is_file()]
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__=='__main__':main()
