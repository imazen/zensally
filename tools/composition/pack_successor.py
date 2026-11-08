#!/usr/bin/env python3
"""Freeze completed local-crop evidence without changing the earlier packet."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

from pack_evidence import copy_file


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path)
    parser.add_argument('localizer', type=Path)
    parser.add_argument('auxiliary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # All declared workers must have completed before taking a frozen copy.
    checks = [
        (args.models/'framelite-ratios/summary.json', 'completed', 32),
        (args.models/'constrained-bank/summary.json', 'requests', 1232),
        (args.localizer/'primary56/summary.json', 'completed', 56),
    ]
    for path, key, expected in checks:
        if json.loads(path.read_text())[key] != expected:
            raise ValueError('required completed evidence missing: ' + str(path))
    for path, count in [(args.localizer/'extent-fits/summary.json', 9),
                        (args.localizer/'extent-giou-fits/summary.json', 9),
                        (args.localizer/'extent-joint-fits/summary.json', 6),
                        (args.localizer/'extent-hint-fits/summary.json', 9),
                        (args.localizer/'focalnet-extent/extent-giou-fits/summary.json', 9)]:
        if len(json.loads(path.read_text())) != count:
            raise ValueError('incomplete registered fit: ' + str(path))
    timing = json.loads((args.auxiliary/'training-timing-isolated-readable.json').read_text())
    if not timing['complete'] or len(timing['invocations']) != 24:
        raise ValueError('corrected timing packet incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, root in [('models', args.models), ('localizer', args.localizer), ('auxiliary', args.auxiliary)]:
        shutil.copytree(root, args.output/name, copy_function=copy_file,
                        ignore=shutil.ignore_patterns('.cache', '__pycache__', 'GOAL_PROGRESS.json'))
    source = args.output/'source'
    files = list(Path('tools/composition').glob('*.py')) + list(Path('tools/composition').glob('*.mjs'))
    files += list(Path('training').glob('*.py'))
    files += [Path('tools/crop-first/fit_profile.py'), Path('tools/crop-first/browser_check.py'), Path('justfile')]
    files += [Path('docs/experiments')/name for name in [
        'COMPOSITION_FEATURE_ABLATION.md', 'SEMANTIC_CROP_SUCCESSOR.md', 'LOCAL_VISION_MODELS.md']]
    files += [Path('benchmarks')/name for name in [
        'local_crop_teachers_2026-11-08.md', 'primary_extents_2026-11-08.md',
        'composition_training_paths_2026-11-08.md', 'composition_features_2026-11-08.md']]
    identities = []
    for file in sorted(set(files)):
        destination = source/file
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_file(file, destination)
        identities.append(dict(path=str(file), sha256=digest(file)))
    logs = args.output/'logs'
    logs.mkdir()
    for file in sorted(Path('/home/lilith/tmp').glob('zensally-composition-*20261108.log')):
        copy_file(file, logs/file.name)
    manifest = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True),
        source_files=identities, packages={name: importlib.metadata.version(name) for name in [
            'torch', 'torchvision', 'numpy', 'transformers', 'timm', 'onnxruntime', 'peft']},
        scope='completed local crop models, weak extent fits, corrected joint fits, profiling and timing; earlier composition packet remains immutable',
        diagnostic_maps_saved=False, paid_model_api_calls=0)
    (args.output/'_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    entries = []
    for file in sorted(args.output.rglob('*')):
        if file.is_file():
            entries.append(dict(path=str(file.relative_to(args.output)), bytes=file.stat().st_size, sha256=digest(file)))
    checksum_path = args.output/'SHA256SUMS.json'
    checksum_path.write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps(dict(packet=str(args.output), members=len(entries),
        payload_bytes=sum(r['bytes'] for r in entries), manifest_sha256=digest(checksum_path))))


if __name__ == '__main__':
    main()
