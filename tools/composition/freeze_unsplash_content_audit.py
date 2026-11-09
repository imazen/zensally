#!/usr/bin/env python3
"""Freeze completed source coverage observations and their bounded previews."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    sources = json.loads((args.root/'sources.json').read_text())
    review = json.loads((args.root/'content-review.json').read_text())
    browser = json.loads((args.root/'browser-unsplash-content.json').read_text())
    if (len(sources['sources'])!=119 or len(review['observations'])!=119 or browser['decoded']!=119
            or browser['exceptions'] or review['test_sources']!=0
            or any(r['role'] not in ['fit','dev'] for r in review['observations'])):
        raise ValueError('complete fit/development audit required')
    screenshots = list((args.root/'screenshots').glob('*.png'))
    if len(screenshots)!=12 or sum(p.stat().st_size for p in screenshots)>24*1024*1024:
        raise ValueError('bounded review capture differs')
    files = [Path('justfile'),Path('tools/crop-first/browser_check.py')]
    files += [Path('tools/composition')/n for n in ['artifact_identity.py','build_unsplash_content_audit.py',
        'unsplash_content_audit.html','unsplash_content_browser_check.mjs',
        'finalize_unsplash_content_audit.py','freeze_unsplash_content_audit.py']]
    files += [Path('docs/experiments/UNSPLASH_CONTENT_AUDIT.md'),Path('benchmarks/unsplash_content_audit_2026-11-09.md')]
    for path in files:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    (args.root/'logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-unsplash-content*20261109.log'):
        shutil.copy2(path,args.root/'logs'/path.name)
    identity = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        registry_sha256=sources['registry_sha256'],review_sha256=sha(args.root/'content-review.json'),
        observation_source='Codex source-preview inspection; unreviewed by a human',
        source_roles_changed=False,training_admission_changed=False,test_sources=0,
        paid_model_api_calls=0,diagnostic_maps_saved=False,face_coverage_certified=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(identity,indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.root)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(args.root.rglob('*')) if p.is_file()]
    sums = args.root/'SHA256SUMS.json'
    sums.write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(dict(members=len(entries),payload_bytes=sum(e['bytes'] for e in entries),manifest_sha256=sha(sums))))


if __name__ == '__main__':
    main()
