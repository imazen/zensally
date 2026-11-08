#!/usr/bin/env python3
"""Freeze all completed detector-family calls, profiles, timings and producers."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    manifest = json.loads((args.root/'manifest.json').read_text())
    coverage = json.loads((args.root/'coverage-analysis.json').read_text())
    profile = json.loads((args.root/'profile-summary.json').read_text())
    timing = json.loads((args.root/'timing-readable.json').read_text())
    browser = json.loads((args.root/'gallery/browser-efficientdet.json').read_text())
    if (coverage['sources'] != 101 or not coverage['original_lite0_rows_exact']
            or len(profile['rows']) != 5 or not timing['complete']
            or len(timing['invocations']) != 24
            or browser['decoded'] != 101 or len(browser['checks']) != 40
            or browser['failures'] or browser['exceptions']):
        raise ValueError('registered complete evidence missing')
    for model in manifest['models']:
        original = Path(model['path'])
        if sha(original) != model['sha256'] or original.stat().st_size != model['bytes']:
            raise ValueError('model identity differs')
        if not original.is_relative_to(args.root):
            shutil.copy2(original, args.root/'models'/original.name)
        rows = json.loads((args.root/'full'/model['id']/'results.json').read_text())['rows']
        if [r['id'] for r in rows] != [r['id'] for r in manifest['images']]:
            raise ValueError('full result ordering differs')
    source = args.root/'source'
    source.mkdir()
    files = [p for parent in ['tools/composition', 'tools/runtime-eval']
             for p in Path(parent).iterdir() if p.is_file()]
    files += [Path(p) for p in ['justfile', 'tools/crop-first/artifacts.py',
        'tools/crop-first/browser_check.py', 'docs/experiments/EFFICIENTDET_LITE_FAMILY.md',
        'benchmarks/efficientdet_family_2026-11-08.md']]
    for path in files:
        target = source/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    logs = args.root/'logs'
    logs.mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-*efficientdet*20261108.log'):
        shutil.copy2(path, logs/path.name)
    identity = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        owned_photo_evidence_manifest_sha256='6594edba891d16c16baa7304058e8008374eb041a52dc6de0545ac9fa42de934',
        owned_photo_inputs_manifest_sha256='1d884d295ed9e958a74d18aaf105aedb896aa493b88eb3542dc60004ccc5e16f',
        model_api_calls=0, quality_model_promoted=False, training_labels_admitted=False,
        face_coverage_certified=False, diagnostic_maps_saved=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.root)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(args.root.rglob('*')) if p.is_file()]
    checksum = args.root/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(members=len(entries), payload_bytes=sum(e['bytes'] for e in entries),
                         manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
