#!/usr/bin/env python3
"""Freeze completed pose-bound witness evidence with exact producers and logs."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    doc = json.loads((args.root/'review/comparison.json').read_text())
    browser = json.loads((args.root/'review/browser-pose-head-review.json').read_text())
    readback = json.loads((args.root/'review/independent-readback.json').read_text())
    default = json.loads((args.root/'default-regression-proof.json').read_text())
    if (doc['summary'] != dict(requests=44, old_crop=38, new_crop=24, changed=34, became_contain=14)
            or browser['failures'] or browser['exceptions'] or browser['decoded'] != 24
            or len(readback['checks']) != 8 or not all(c['owned_png_rgb_exact'] for c in readback['checks'])
            or not default['all_json_exact'] or default['compared_files'] != 103):
        raise ValueError('registered completion/guard evidence differs')
    source = args.root/'source'
    source.mkdir()
    files = [f for f in Path('tools/composition').iterdir() if f.is_file()]
    files += [Path('tools/crop-first')/name for name in ['artifacts.py', 'browser_check.py']]
    files += [Path('justfile'), Path('docs/experiments/POSE_HEAD_BOUNDS.md'),
              Path('benchmarks/pose_head_screen_2026-11-08.md')]
    for file in files:
        target = source/file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    logs = args.root/'logs'
    logs.mkdir()
    for file in Path('/home/lilith/tmp').glob('zensally-composition-pose-head-*20261108.log'):
        shutil.copy2(file, logs/file.name)
    identity = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        source_photo_evidence_manifest_sha256='6594edba891d16c16baa7304058e8008374eb041a52dc6de0545ac9fa42de934',
        cached_runtime_manifest_sha256='a49bbedcd4894ed7a7dd0c5a9f6b0f3a31647aba8f4e02d5675caa95e2e4d04e',
        quality_model_promoted=False, training_eligible=False, diagnostic_maps_saved=False,
        paid_model_api_calls=0, face_coverage_certified=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
    entries = [dict(path=str(f.relative_to(args.root)), bytes=f.stat().st_size, sha256=sha(f))
               for f in sorted(args.root.rglob('*')) if f.is_file()]
    checksum = args.root/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(members=len(entries), payload_bytes=sum(e['bytes'] for e in entries),
                          manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
