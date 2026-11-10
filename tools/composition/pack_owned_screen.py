#!/usr/bin/env python3
"""Freeze a completed review-only photo screen and its renderer correction."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from artifact_identity import sha
from pack_evidence import copy_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['screen', 'acquired', 'earlier_galleries', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.screen.resolve()):
        raise ValueError('packet must be outside its source tree')
    def read(path):
        return json.loads(path.read_text())
    inputs = read(args.screen/'inputs.json')
    numerical = read(args.screen/'cached-numerics-optimized/summary.json')
    roi = read(args.screen/'person-roi-results/summary.json')
    before = read(args.screen/'crop-clipping-before-stable.json')
    after = read(args.screen/'crop-clipping-after-stable.json')
    if (len(inputs['images']) != 101 or numerical['requests'] != 2222
            or numerical['choice_changes'] or numerical['exact_score_arrays'] != 1111
            or numerical['exact_importance_arrays'] != 1111 or roi['calls'] != 54
            or before['failed_samples'] != 4 or after['failed_samples'] != 0):
        raise ValueError('required completed gate differs')
    for name in ['native-numerics-dispatched', 'native-numerics-scalar']:
        for mode in read(args.screen/name/'summary.json').values():
            if not mode['passed'] or mode['requests'] != 2222:
                raise ValueError('native gate incomplete')
    input_checksum = args.acquired/'SHA256SUMS.json'
    input_digest = '1d884d295ed9e958a74d18aaf105aedb896aa493b88eb3542dc60004ccc5e16f'
    if sha(input_checksum) != input_digest:
        raise ValueError('frozen acquired input identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.screen, args.output/'screen', copy_function=copy_file,
                    ignore=shutil.ignore_patterns('__pycache__', 'GOAL_PROGRESS.json'))
    copy_file(input_checksum, args.output/'ACQUIRED_INPUT_SHA256SUMS.json')
    for name in ['gallery', 'gallery-semantic-v2', 'gallery-constrained']:
        shutil.copytree(args.earlier_galleries/name, args.output/'corrected-earlier-viewers'/name,
                        copy_function=copy_file)
    files = [f for f in Path('tools/composition').iterdir() if f.is_file()]
    files += [Path('tools/crop-first')/name for name in ['browser_check.py', 'artifacts.py',
        'acquire_current.py', 'acquisition_oracle.py', 'corpus.py', 'photo_focus.py']]
    files += [Path('docs/experiments')/name for name in ['OWNED_PHOTO_SCREEN.md', 'PERSON_ROI_REFINEMENT.md']]
    files += [Path('justfile'), Path('DATA_PROVENANCE.md'), Path('benchmarks/owned_photo_screen_2026-11-08.md')]
    for file in sorted(files):
        target = args.output/'source'/file
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(file, target)
    logs = args.output/'logs'
    logs.mkdir()
    for file in sorted(Path('/home/lilith/tmp').glob('zensally-composition-*20261108.log')):
        if any(key in file.name for key in ['owned-photo', 'person-roi', 'clipped-gallery', 'roi-unit']):
            copy_file(file, logs/file.name)
    environment = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        acquired_input_manifest_sha256=input_digest, acquired_input_members=1012,
        acquired_input_local=str(args.acquired),
        acquired_input_nas='/mnt/tower/input/zensally/owned-photo-screen/2026-11-08/',
        external_cached_runtime_manifest_sha256='a49bbedcd4894ed7a7dd0c5a9f6b0f3a31647aba8f4e02d5675caa95e2e4d04e',
        external_model_inventory_manifest_sha256='4a58cd6aa5aa966bbaba3cd02506d7cdba545bde9371974421242c0effd4689b',
        new_training_labels=False, source_roles_changed=False, diagnostic_maps_saved=False,
        paid_model_api_calls=0, quality_model_promoted=False,
        scope='101 review-only photos, arithmetic gates, detector witnesses, two-source ROI probe and SVG correction')
    (args.output/'_MANIFEST.json').write_text(json.dumps(environment, indent=2)+'\n')
    entries = [dict(path=str(file.relative_to(args.output)), bytes=file.stat().st_size,
                    sha256=sha(file)) for file in sorted(args.output.rglob('*')) if file.is_file()]
    checksum = args.output/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(packet=str(args.output), members=len(entries),
        payload_bytes=sum(e['bytes'] for e in entries), manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
