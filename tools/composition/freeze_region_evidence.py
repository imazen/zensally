#!/usr/bin/env python3
"""Seal completed region/placement evidence without rewriting original results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from artifacts import sha,verify


def read(root,name):
    return json.loads((root/name).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['regions','native','tiles','projection','padded','review','review-initial'])
    p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    if (root/'SHA256SUMS.json').exists():raise ValueError('sealed packet must not be rewritten')
    if a.kind=='regions':
        result=read(root,'analysis/summary.json')
        if result['completed_fits']!=45 or not result['all_selected_native_replays_exact'] or not result['repeated_geometry_controls_bit_identical']:raise ValueError('complete matched native controls required')
        for variant,rows in result['identities'].items():
            progress=read(root,variant+'/fits/PROGRESS.json')
            if progress['stage']!='complete' or progress['completed']!=9 or progress['pid'] is not None or len(rows)!=9:raise ValueError('fit worker not complete')
            for row in rows:
                if sha(root/variant/'fits'/f"{row['arm']}-seed{row['seed']}"/'best.pt')!=row['weights_sha256']:raise ValueError('selected state differs')
        if any(row['max_edge_128_grid']>3 for row in result['cpu_gpu_native_checks']):raise ValueError('numerical gate differs')
        patterns=['zensally-region-context-*-20261109.log']
    elif a.kind=='native':
        result=read(root,'analysis/summary.json')
        if result['newly_evaluated_calls']!=144 or len(result['local_scopes'])!=2 or read(root,'guard-audit/summary.json')['model_promoted']:raise ValueError('native evaluation incomplete')
        patterns=['zensally-subject-native-faces-*-20261109.log']
    elif a.kind=='tiles':
        result=read(root,'analysis/summary.json');guard=read(root,'guard-audit/summary.json');projection=read(root,'projection25/summary.json')
        if result['newly_evaluated_calls']!=288 or len(result['local_scopes'])!=2 or guard['model_promoted'] or not projection['numerical_gate_passed'] or projection['numerical_failures']:raise ValueError('complete tile/guard/native projection results required')
        if len(read(root,'manifest.json')['images'])!=288:raise ValueError('all fixed tiles required')
        patterns=['zensally-subject-tiled-faces-*-20261109.log']
    elif a.kind in ['projection','padded']:
        roots=[root] if a.kind=='projection' else [root/'p10',root/'p25']
        for r in roots:
            result=read(r,'summary.json');progress=read(r,'PROGRESS.json')
            if not result['numerical_gate_passed'] or result['numerical_failures'] or progress['stage']!='complete' or progress['pid'] is not None:raise ValueError('complete native placement audit required')
        patterns=['zensally-subject-origin-projection-*-20261109.log']
    elif a.kind=='review':
        browser=read(root,'browser-miap-rank.json');selection=read(root,'selections.json')
        if browser['checks']!=704 or browser['failures'] or browser['exceptions'] or len(selection['sources'])!=8:raise ValueError('bounded browser review incomplete')
        result=dict(api_calls=0,model_promoted=False,test_inference=False,face_coverage_certified=False)
        patterns=['zensally-subject-origin-review-*-20261109.log']
    else:
        selection=read(root,'selections.json')
        if len(selection['sources'])!=8 or any(len(row['tasks'])!=22 for row in selection['sources']):raise ValueError('initial generated viewer is incomplete')
        result=dict(api_calls=0,model_promoted=False,test_inference=False,face_coverage_certified=False)
        patterns=[]
    if any(result.get(key,False) for key in ['api_calls','model_promoted','test_inference','face_coverage_certified']):raise ValueError('unexpected promotion, inference or spending')
    names=['justfile','training/direct_crop.py','training/region_context.py','training/test_region_context.py',
        'training/checkpoint_snapshot.py','training/test_checkpoint_snapshot.py','tools/composition/cache_region_context.py',
        'tools/composition/fit_direct_crop.py','tools/composition/analyze_region_context.py',
        'tools/composition/subject_origin_projection.py','tools/composition/test_subject_origin_projection.py',
        'tools/composition/audit_subject_origin_projection.py','tools/composition/origin_projection_gallery.py',
        'tools/composition/subject_native_faces.py','tools/composition/audit_local_crop_guards.py',
        'tools/composition/subject_policy.py','tools/composition/artifact_identity.py',
        'tools/composition/miap_rank_browser_check.mjs','tools/crop-first/browser_check.py',
        'tools/crop-first/subject_geometry.py','tools/crop-first/artifacts.py',
        'docs/experiments/REGION_CONTEXT_SCOUT.md','benchmarks/region_context_2026-11-09.md',
        'benchmarks/subject_origin_projection_2026-11-09.md','tools/composition/freeze_region_evidence.py']
    if a.kind=='tiles':names+=['tools/composition/subject_tiled_faces.py','tools/composition/test_subject_tiled_faces.py',
        'docs/experiments/TILED_FACE_CONTROL.md','benchmarks/subject_tiled_faces_2026-11-09.md']
    for name in names:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    asrun={'region-fit':('7d9dfbde','tools/composition/fit_direct_crop.py'),
        'initial-projection':('091132a7','tools/composition/audit_subject_origin_projection.py'),
        'native-projection':('4ada7ff4','tools/composition/audit_subject_origin_projection.py'),
        'padding-projection':('d8abf968','tools/composition/audit_subject_origin_projection.py'),
        'initial-viewer':('9de7b541','tools/composition/origin_projection_gallery.py'),
        'status-viewer':('c95a41fd','tools/composition/origin_projection_gallery.py'),
        'padding-viewer':('03759771','tools/composition/origin_projection_gallery.py')}
    if a.kind=='tiles':asrun['tile-producer']=('85411f55','tools/composition/subject_tiled_faces.py')
    for name,(revision,path) in asrun.items():
        target=root/'source/as-run'/f'{name}.py';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','show',revision+':'+path]))
    (root/'logs').mkdir()
    for pattern in patterns:
        for path in (Path.home()/'tmp').glob(pattern):shutil.copy2(path,root/'logs'/path.name)
    manifest=dict(kind=a.kind,api_calls=0,model_promoted=False,test_inference=False,source_roles_changed=False,
        face_coverage_certified=False,human_artistic_gold=False,source_revision=subprocess.check_output(
            ['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),as_run_sources=asrun,
        parent_packets=dict(direct_crop='65a6be0d8cdac4bcea97900f352afeab00611cc025dbeb39e70bc2f75697a046',
            face_cues='d69e2460bfe6c0fd6a680969a35eb0828bb697049c22fb81cb64cfb65755215a',
            bank_controls='fb73334275761c51483b0fb77c3a63e80c80bb290404dcb403ea8483b859beb6',
            followup='90b7c63a38b247bad5cd8a55bf15c7b8b05e04aeefc348dbf388fc0c7c2000d3',
            runtime_models='b6bf576e6f3539f1ae9d6231e78ded484c0a35b35d980dd7eb101549be063230'))
    if a.kind=='tiles':manifest['parent_packets'].update(native_faces='20adfd7777f535cce3ad328992b1f56ae76f7b0b87048f9a299bcdc98789441f',
        region_context='9b68ca73fb8e4fb27764290d5b5d7243d8323b52ab2df1fd4787de6c3726d610')
    if a.kind.startswith('review'):
        manifest.update(retained_screenshots=10 if a.kind=='review' else 0,visually_reviewed_screenshots=4 if a.kind=='review' else 0,
            initial_status_display_unreliable=a.kind=='review-initial',browser_checks=704 if a.kind=='review' else 0)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(root,entries)
    print(json.dumps(dict(kind=a.kind,root=str(root),members=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
