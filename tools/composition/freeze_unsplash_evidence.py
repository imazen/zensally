#!/usr/bin/env python3
"""Freeze qualified owned previews and the separate unlabeled detector study."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preparation', type=Path)
    parser.add_argument('detectors', type=Path)
    args = parser.parse_args()
    views = json.loads((args.preparation/'working-previews/manifest.json').read_text())
    browser = json.loads((args.detectors/'gallery/browser-efficientdet.json').read_text())
    old = json.loads((args.detectors/'old-gallery-regression/browser-efficientdet.json').read_text())
    results = json.loads((args.detectors/'full/summary.json').read_text())
    if (len(views['rows']) != 144 or not all(r['independent_png_rgb_exact'] for r in views['rows'])
            or browser['decoded'] != 144 or old['decoded'] != 101
            or browser['failures'] or browser['exceptions'] or old['failures'] or old['exceptions']
            or len(results) != 5 or any(r['returncode'] for r in results)):
        raise ValueError('complete preview/detector/browser evidence required')
    for row in views['rows']:
        if sha(Path(row['png'])) != row['png_sha256'] or sha(Path(row['rgb'])) != row['rgb_sha256']:
            raise ValueError('qualified owned pixel artifact changed')
    commit = subprocess.check_output(['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip()
    for root, report, patterns in [
            (args.preparation, 'unsplash_preparation_2026-11-08.md', ['*unsplash*', '*corpus-preview*', '*owned-probe*']),
            (args.detectors, 'unsplash_detectors_2026-11-08.md', ['*unsplash-detectors*', '*efficientdet-old-gallery*'])]:
        if (root/'SHA256SUMS.json').exists():
            raise ValueError('packet already frozen')
        files = [p for parent in ['tools/composition', 'tools/runtime-eval'] for p in Path(parent).iterdir() if p.is_file()]
        files += list(Path('tools/corpus-preview').glob('Cargo.*'))
        files += [Path('tools/corpus-preview/src/main.rs'), Path('tools/corpus-preview/README.md'),
                  Path('tools/crop-png/src/main.rs'), Path('tools/crop-png/Cargo.toml'),
                  Path('tools/crop-first/artifacts.py'), Path('tools/crop-first/browser_check.py'),
                  Path('justfile'), Path('benchmarks')/report]
        files += [Path('docs/experiments')/name for name in ['UNSPLASH_SCREEN.md', 'UNSPLASH_DETECTORS.md']]
        for path in files:
            target = root/'source'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        logs = root/'logs'
        logs.mkdir()
        for pattern in patterns:
            for path in Path('/home/lilith/tmp').glob('zensally-'+pattern+'20261108.log'):
                if not (logs/path.name).exists():
                    shutil.copy2(path, logs/path.name)
        identity = dict(build_commit=commit,
            source_input_packet_sha256='45ef06063bf431d90c2e41db9e519133aa323ac142976843c0c3092bb9bb6e64',
            detector_weight_packet_sha256='43b0cc63ef8bd17afdc48884cf029ddf791462c18ae87bbd2f09a0e44d9d836d',
            working_manifest_sha256=sha(args.preparation/'working-previews/manifest.json'),
            source_roles_assigned=False, training_labels_admitted=False,
            face_coverage_certified=False, diagnostic_maps_saved=False, paid_model_api_calls=0)
        (root/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
        entries = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(root.rglob('*')) if p.is_file()]
        checksum = root/'SHA256SUMS.json'
        checksum.write_text(json.dumps(entries, indent=2)+'\n')
        print(json.dumps(dict(root=str(root), members=len(entries),
            payload_bytes=sum(p['bytes'] for p in entries), manifest_sha256=sha(checksum))), flush=True)


if __name__ == '__main__':
    main()
