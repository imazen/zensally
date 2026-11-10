#!/usr/bin/env python3
"""Freeze completed preference fits, exact producers and existing witness pixels."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from artifact_identity import sha
from pack_evidence import copy_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['mixed', 'photo', 'rendered', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in [args.mixed, args.photo]):
        raise ValueError('packet must be outside source trees')
    expected = {(arm, seed) for arm in ['visual', 'geometry', 'permuted'] for seed in [31, 137, 911]}
    for root, families in [(args.mixed, 16), (args.photo, 5)]:
        summary = json.loads((root/'fits/summary.json').read_text())
        progress = json.loads((root/'fits/progress.json').read_text())
        analysis = json.loads((root/'analysis/summary.json').read_text())
        if (len(summary) != 9 or {(r['arm'], r['seed']) for r in summary} != expected
                or progress['stage'] != 'complete' or progress['completed_runs'] != 9
                or analysis['families'] != families or len(analysis['checkpoints']) != 12):
            raise ValueError('registered preference study is incomplete: '+str(root))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, root in [('mixed', args.mixed), ('photo', args.photo)]:
        shutil.copytree(root, args.output/name, copy_function=copy_file,
                        ignore=shutil.ignore_patterns('__pycache__', 'GOAL_PROGRESS.json'))
    cohort = json.loads((args.photo/'cohort.json').read_text())
    for entry in cohort['packets']:
        original = Path(entry['path'])
        if sha(original) != entry['sha256']:
            raise ValueError('source metadata changed since cohort selection')
        target = args.output/'source-packets'/original.parent.name/'packet.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(original, target)
    gallery = json.loads((args.photo/'gallery/selections.json').read_text())
    pixels = set()
    for source in gallery['sources']:
        pixels.add((source['id'], 'source.png'))
        for task in source['tasks']:
            for choice in task['choices']:
                pixels.add((source['id'], Path(urlparse(choice['image']).path).name))
    for source, name in sorted(pixels):
        target = args.output/'witness-pixels'/source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(args.rendered/source/name, target)
    files = list(Path('tools/composition').glob('*.py'))
    files += list(Path('tools/composition').glob('preference_*.mjs'))
    files += [Path('tools/composition/preference_gallery.html'), Path('tools/crop-first/browser_check.py'),
              Path('tools/crop-first/photo_focus.py'), Path('justfile')]
    files += list(Path('training').glob('*.py'))
    files += [Path('docs/experiments')/name for name in ['FOCAL_PREFERENCE_HEAD.md', 'PHOTO_PREFERENCE_HEAD.md']]
    files += [Path('benchmarks/focal_preference_heads_2026-11-08.md')]
    for file in sorted(set(files)):
        target = args.output/'source'/file
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(file, target)
    # The original mixed fits predate the opt-in photographic-cohort arguments.
    producers = [('28d59440f552', 'tools/composition/fit_focal_preferences.py'),
                 ('115ef910633e', 'tools/composition/analyze_focal_preferences.py'),
                 ('b1dfacfa80ab', 'tools/composition/fit_focal_preferences.py'),
                 ('b1dfacfa80ab', 'tools/composition/analyze_focal_preferences.py')]
    for revision, file in producers:
        target = args.output/'producer-source'/revision/file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', revision+':'+file]))
    logs = args.output/'logs'
    logs.mkdir()
    for file in sorted(Path('/home/lilith/tmp').glob('zensally-composition-*preference*20261108.log')):
        copy_file(file, logs/file.name)
    for name in ['zensally-composition-focal-head-unit-20261108.log', 'zensally-composition-all-tests-20261108.log']:
        file = Path('/home/lilith/tmp')/name
        copy_file(file, logs/name)
    manifest = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'timm']},
        witness_pixels=len(pixels), source_roles_changed=False, diagnostic_maps_saved=False,
        paid_model_api_calls=0, quality_model_promoted=False,
        external_encoder='pinned FocalNet assets retained in the earlier successor teacher packet',
        external_encoder_packet_manifest_sha256='378ec058d53ade02815ca355186578e295b435286e26f51e34f851cf528fc225')
    (args.output/'_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    entries = [dict(path=str(file.relative_to(args.output)), bytes=file.stat().st_size,
                    sha256=sha(file)) for file in sorted(args.output.rglob('*')) if file.is_file()]
    checksum = args.output/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(packet=str(args.output), members=len(entries),
        payload_bytes=sum(r['bytes'] for r in entries), manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
