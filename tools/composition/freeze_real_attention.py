#!/usr/bin/env python3
"""Seal completed teacher inputs/features while keeping unstarted fits explicit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from artifacts import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    if (root/'SHA256SUMS.json').exists():raise ValueError('sealed packet must not be rewritten')
    status=json.loads((root/'cache/PROGRESS.json').read_text());summary=json.loads((root/'cache/summary.json').read_text())
    if status['stage']!='complete' or status['completed']!=72 or status['pid'] is not None or sha(root/'cache/cache.pt')!=summary['cache_sha256'] or not summary['initial_replay_exact']:raise ValueError('complete exact candidate cache required')
    if (root/'fits').exists():raise ValueError('this freeze is explicitly pre-fit')
    completion=json.loads((root/'teacher-completion-verification.json').read_text())
    if any(r['worker_live'] or not r['summary_present'] for r in completion['observations']):raise ValueError('teacher terminal states not verified')
    inputs=json.loads((root/'inputs.json').read_text())
    for row in inputs['images']:
        for kind in ['gaze','scan']:
            if sha(Path(row[kind]['path']))!=row[kind]['sha256']:raise ValueError('owned teacher input differs')
            result=json.loads((root/kind/(row['id']+'.json')).read_text())
            if any(result[k]!=row[k] for k in ['id','role','family_id','rgb_sha256']):raise ValueError('teacher sidecar identity differs')
    names=['justfile','tools/composition/prepare_real_attention.py','tools/composition/cache_real_attention.py','tools/composition/attention.py',
        'tools/composition/teacher_models.py','tools/composition/test_attention_heads.py','training/composition_features.py',
        'training/attention_scalar_head.py','training/test_attention_scalar_head.py','training/checkpoint_snapshot.py',
        'docs/experiments/REAL_ATTENTION_CONTROL.md','docs/experiments/LABEL_QUALITY_AUDIT.md',
        'tools/composition/artifact_identity.py','tools/crop-first/artifacts.py','tools/composition/freeze_real_attention.py']
    for name in names:
        path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,path)
    path=root/'source/as-run';path.mkdir()
    for revision,name,label in [('81d22fc9','tools/composition/attention.py','attention-producer'),('a9053554','tools/composition/cache_real_attention.py','initial-cache-producer')]:
        (path/(label+'.py')).write_bytes(subprocess.check_output(['git','show',revision+':'+name]))
    (root/'logs').mkdir()
    for path in (Path.home()/'tmp').glob('zensally-real-attention-*-20261109.log'):shutil.copy2(path,root/'logs'/path.name)
    manifest=dict(stage='teacher sidecars and frozen hidden cache complete; registered fits not started',registered_fits=15,completed_fits=0,
        api_calls=0,model_promoted=False,test_inference=False,face_coverage_certified=False,source_roles_changed=False,
        paid_label_audit_is_separate=True,source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True),
        parent_packets=inputs['parent_packets'],teacher_weights_are_verified_parent_pointers=True)
    (root/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'SHA256SUMS.json').write_text(json.dumps(entries,indent=2)+'\n');verify(root,entries)
    print(json.dumps(dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),manifest_sha256=sha(root/'SHA256SUMS.json'))))


if __name__=='__main__':main()
