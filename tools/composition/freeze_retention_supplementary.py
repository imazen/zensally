#!/usr/bin/env python3
"""Freeze supplementary crop failures, native geometry and bounded rendered review."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if (a.root/'SHA256SUMS.json').exists():raise ValueError('packet already frozen')
    if sha(a.root/'plan.json')!='35f01f5ca0c3be192980ef9d7c749e1bbb346c6cdaa80e0cfaa096e86c43ec8a':raise ValueError('pre-prediction plan differs')
    progress=json.loads((a.root/'progress.json').read_text())
    result=json.loads((a.root/'analysis.json').read_text())
    browser=json.loads((a.root/'review/browser-retention-review.json').read_text())
    if (progress['pid'] is not None or progress['completed_models']!=9 or result['reference_boxes']!=23
        or result['original_scorer_max_error']!=0 or browser['checks']!=2970 or browser['decoded']!=15 or browser['exceptions']):raise ValueError('complete numerical/source/browser evidence required')
    (a.root/'selected-states').mkdir()
    states=[]
    for model in result['models']:
        path=Path(model['path'])
        if sha(path)!=model['sha256']:raise ValueError('frozen selection changed')
        target=a.root/'selected-states'/(model['arm']+'-'+str(model['seed'])+'.pt');shutil.copy2(path,target)
        states.append(dict(original=str(path),copy=str(target.relative_to(a.root)),sha256=sha(target)))
    paths=[Path('justfile'),Path('docs/experiments/RETENTION_SUPPLEMENTARY.md'),Path('benchmarks/retention_supplementary_2026-11-09.md'),Path('tools/crop-first/browser_check.py')]
    paths+=[Path('tools/composition')/n for n in ['artifact_identity.py','plan_retention_supplementary.py',
        'evaluate_retention_supplementary.py','freeze_retention_supplementary.py','build_retention_review.py',
        'retention_review.html','retention_review_browser_check.mjs','fit_photo_retention.py',
        'analyze_unsplash_ranks.py','constrained_bank.py','audit_crop_feasibility.py','region_retention.py','fit_extents.py']]
    paths+=[Path('training')/n for n in ['local_rank_distillation.py','crop_fit.py','crop_ranker.py','crop_regions.py','model_v3ds.py']]
    producers={}
    for path in paths:
        target=a.root/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        producers[str(path)]=subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (a.root/'logs').mkdir()
    for pattern in ['zensally-retention-supplementary*20261109.log','zensally-retention-review*20261109.log']:
        for path in Path('/home/lilith/tmp').glob(pattern):shutil.copy2(path,a.root/'logs'/path.name)
    manifest=dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        source_producers=producers,selected_states=states,models=9,supplementary_families=15,weak_reference_boxes=23,
        parent_retention_packet_sha256='3e603f65c94814b360fe8bee64484cffe07c39024ef32a4dda6e062c85a1698a',
        parent_reference_packet_sha256='9fe44aa3645d379ab1ab925731f4710674f84b03171908026b1b3210a553e9f4',
        parent_source_packet_sha256='20c5c0b330b12357b5e76cfba85f356992b0ca7ee35eea59e3594e6596512d04',
        new_gradients=False,new_teacher_calls=0,new_test_inference=False,paid_model_api_calls=0,
        source_roles_changed=False,model_promoted=False,face_coverage_certified=False,
        human_artistic_quality_measured=False,diagnostic_maps_saved=False)
    (a.root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(a.root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.root.rglob('*')) if p.is_file()]
    if sum(r['bytes'] for r in entries)>256*1024**2:raise ValueError('256MiB packet bound exceeded; preserve files')
    sums=a.root/'SHA256SUMS.json'
    with sums.open('x') as stream:stream.write(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(sums))))


if __name__=='__main__':main()
