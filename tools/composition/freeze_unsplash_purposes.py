#!/usr/bin/env python3
"""Freeze the completed bounded purpose-policy comparison and its producers."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('export', type=Path)
    args = parser.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    summary = json.loads((args.root/'evaluation/comparison.json').read_text())['summary']
    browser = json.loads((args.root/'evaluation/browser-unsplash-purpose.json').read_text())
    if (summary != dict(requests=330, research_crop=270, unconditioned_region_violations=144,
            selected_region_violations=0, origin_recoveries=32, maximal_fallbacks=6, contain=60)
            or browser['requests'] != 330 or browser['crops'] != 600 or browser['padded'] != 60
            or browser['failures'] or browser['exceptions']):
        raise ValueError('registered completion evidence differs')
    files = [Path('tools/composition')/name for name in ['artifact_identity.py', 'constrained_bank.py',
        'audit_crop_feasibility.py', 'test_constrained_geometry.py', 'unsplash_purpose_witnesses.py',
        'evaluate_unsplash_purposes.py', 'build_unsplash_purpose_gallery.py', 'unsplash_purpose_gallery.html',
        'unsplash_purpose_browser_check.mjs', 'crop_clip.js', 'freeze_unsplash_purposes.py']]
    files += [Path('justfile'), Path('tools/crop-first/browser_check.py'), Path('tools/crop-first/artifacts.py'),
              Path('docs/experiments/UNSPLASH_PURPOSE_WITNESSES.md'), Path('docs/experiments/CROP_PURPOSES.md'),
              Path('benchmarks/unsplash_purpose_witnesses_2026-11-09.md')]
    for path in files:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (args.root/'model').mkdir()
    for name in ['encoder.onnx', 'query.onnx', 'qualification.json']:
        shutil.copy2(args.export/name, args.root/'model'/name)
    (args.root/'logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-unsplash-purpose*20261109.log'):
        shutil.copy2(path, args.root/'logs'/path.name)
    identity = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        annotation_sha256=sha(args.root/'annotations.json'),
        original_rank_packet_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        student_runtime_packet_sha256='a84bdeda2cdcbbebcd049f33223bb8ab1f5244d43adcf84ee29289b2ab0e3ae6',
        source_roles_changed=False, model_has_purpose_input=False, training_updated=False,
        human_artistic_quality_measured=False, face_coverage_certified=False,
        model_promoted=False, new_test_inference=False, paid_model_api_calls=0, diagnostic_maps_saved=False)
    (args.root/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.root)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(args.root.rglob('*')) if p.is_file()]
    path = args.root/'SHA256SUMS.json'
    path.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(members=len(entries), payload_bytes=sum(v['bytes'] for v in entries), manifest_sha256=sha(path))))


if __name__ == '__main__':
    main()
