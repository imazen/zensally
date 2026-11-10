#!/usr/bin/env python3
"""Freeze completed cached/native runtime evidence without editing earlier packets."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

from artifact_identity import sha
from pack_evidence import copy_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('zenbench', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.runtime.resolve()):
        raise ValueError('packet must be outside its source tree')
    for name in ['timing', 'timing-mixed', 'timing-spinning', 'query-timing', 'analysis-timing']:
        data = json.loads((args.runtime/(name+'-readable.json')).read_text())
        if not data['complete'] or any(r['status'] != 'completed' for r in data['invocations']):
            raise ValueError('required timing did not complete: '+name)
    for name in ['numerics-bits-disabled', 'numerics-bits-optimized',
                 'numerics-bits-mixed-t4q1', 'numerics-bits-no-spin-t4q4']:
        data = json.loads((args.runtime/name/'summary.json').read_text())
        if (data['requests'], data['exact_score_arrays'], data['exact_importance_arrays']) != (1232, 616, 616):
            raise ValueError('required bit-parity gate missing: '+name)
    for name in ['native-gate-dispatched', 'native-gate-scalar']:
        data = json.loads((args.runtime/name/'summary.json').read_text())
        if any(not mode['passed'] or mode['requests'] != 1232 for mode in data.values()):
            raise ValueError('required native gate missing: '+name)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.runtime, args.output/'runtime', copy_function=copy_file,
                    ignore=shutil.ignore_patterns('__pycache__', 'GOAL_PROGRESS.json'))
    copy_file(args.zenbench, args.output/'zenbench-0.1.10')
    source = args.output/'source'
    files = list(Path('tools/composition').glob('*.py'))
    files += [Path('tools/query-native')/name for name in [
        'Cargo.toml', 'Cargo.lock', 'build.rs', 'src/main.rs', 'README.md']]
    files += [Path('justfile'), Path('benchmarks/cached_crop_runtime_2026-11-08.md')]
    identities = []
    for file in sorted(files):
        destination = source/file
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_file(file, destination)
        identities.append(dict(path=str(file), sha256=sha(file)))
    logs = args.output/'logs'
    logs.mkdir()
    for file in sorted(Path('/home/lilith/tmp').glob('zensally-composition-*20261108.log')):
        copy_file(file, logs/file.name)
    manifest = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        source_files=identities, packages={name: importlib.metadata.version(name)
            for name in ['numpy', 'onnx', 'onnxruntime']},
        scope='cached FocalNet and embedded native query experiments; earlier teacher/fit packets unchanged',
        diagnostic_maps_saved=False, new_training_labels=False, paid_model_api_calls=0)
    (args.output/'_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    entries = [dict(path=str(file.relative_to(args.output)), bytes=file.stat().st_size,
                    sha256=sha(file)) for file in sorted(args.output.rglob('*')) if file.is_file()]
    checksum = args.output/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(packet=str(args.output), members=len(entries),
        payload_bytes=sum(r['bytes'] for r in entries), manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
