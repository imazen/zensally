#!/usr/bin/env python3
"""Freeze both completed ROI encoder studies, dereferencing checkpoint aliases."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['synthetic', 'small', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    comparison = json.loads((args.small/'encoder-comparison.json').read_text())
    if not all(comparison['geometry_control_exact'].values()):
        raise ValueError('matched geometry control required')
    for root, fits in [(args.synthetic, 'head-fits'), (args.small, 'fits')]:
        progress = json.loads((root/fits/'progress.json').read_text())
        runs = json.loads((root/fits/'summary.json').read_text())
        if (progress['completed_runs'] != 9 or progress['stage'] != 'complete'
                or {(r['arm'], r['seed']) for r in runs} !=
                   {(a, s) for a in ['image', 'geometry', 'permuted'] for s in [31, 137, 911]}):
            raise ValueError('registered nine-run study incomplete')
    oracle = json.loads((args.synthetic/'input-oracle.json').read_text())
    if len(oracle['checks']) != 117:
        raise ValueError('complete117-view readback required')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, root in [('synthetic', args.synthetic), ('small', args.small)]:
        # copytree's default symlinks=False materializes latest/best aliases;
        # the frozen packet remains valid after copying without symlink support.
        shutil.copytree(root, args.output/name, symlinks=False)
    if any(p.is_symlink() for p in args.output.rglob('*')):
        raise ValueError('packet must have no unresolved aliases')
    files = [p for parent in ['tools/composition', 'training'] for p in Path(parent).glob('*.py')]
    files += [Path('tools/crop-first/artifacts.py'), Path('justfile'),
              Path('benchmarks/synthetic_person_rois_2026-11-08.md')]
    files += [Path('docs/experiments')/name for name in [
        'SYNTHETIC_PERSON_ROIS.md', 'ROI_HEAD_FIT.md', 'ROI_HEAD_SMALL_ENCODER.md']]
    for path in files:
        target = args.output/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    logs = args.output/'logs'
    logs.mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-*roi*20261108.log'):
        shutil.copy2(path, logs/path.name)
    identity = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        external_encoder_packet_manifest_sha256='378ec058d53ade02815ca355186578e295b435286e26f51e34f851cf528fc225',
        synthetic_source_packet_manifest_sha256='d19651ba88ace98b07175459520b683879cc9bea73db657b490908a67edb5c76',
        completed_fits=18, paid_model_api_calls=0, diagnostic_maps_saved=False,
        source_roles_changed=False, canonical_holdouts_opened=False,
        quality_model_promoted=False, face_coverage_certified=False)
    (args.output/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.output)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(args.output.rglob('*')) if p.is_file()]
    manifest = args.output/'SHA256SUMS.json'
    manifest.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(members=len(entries), payload_bytes=sum(p['bytes'] for p in entries),
                         manifest_sha256=sha(manifest))))


if __name__ == '__main__':
    main()
