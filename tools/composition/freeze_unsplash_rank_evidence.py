#!/usr/bin/env python3
"""Freeze finished relationship, role, fit/TEST and profiler packets in place."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_base', type=Path)
    args = parser.parse_args()
    base = args.output_base
    rank = base/'unsplash-rank-distillation/2026-11-08'
    roles = base/'unsplash-roles/2026-11-08'
    relationships = base/'unsplash-relationships/2026-11-08'
    profiles = [base/('unsplash-rank-training-profile/'+name) for name in ['2026-11-08', '2026-11-08-scoped']]
    if sha(roles/'registry.json') != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('frozen role identity differs')
    for path, digest in [('cache/cache.pt', '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),
                         ('test-cache/cache.pt', '2f5a2d2fc1b61ce34d5181746bd12d46ad9973ae19f2ee0db4b1e7c2c5f16f6f'),
                         ('analysis/test-plan.json', '6358383b2b7b2e1a3386334168fd9b457c9c2c17b2e37b96b74b003520696f45')]:
        if sha(rank/path) != digest:
            raise ValueError('frozen rank-study input differs')
    progress = json.loads((rank/'fits/progress.json').read_text())
    evaluation = json.loads((rank/'test-evaluation/summary.json').read_text())
    audit = json.loads((relationships/'summary.json').read_text())
    if (progress['completed_runs'] != 12 or progress['stage'] != 'complete'
            or len(evaluation['models']) != 12 or evaluation['model_reselected']
            or audit['pairs'] != 10296 or not audit['calibration_exact']):
        raise ValueError('registered study is incomplete')
    for binding in json.loads((rank/'analysis/test-plan.json').read_text())['models']:
        if sha(Path(binding['path'])) != binding['sha256']:
            raise ValueError('frozen selected model differs')
    scope = json.loads((profiles[1]/'TARGET_SCOPE.json').read_text())
    if scope['memory_max_bytes'] != 4294967296:
        raise ValueError('actual profiling target cap is unverified')
    for root in profiles:
        report = json.loads((root/'PROFILE_COMPLETION.json').read_text())
        if not report['completed'] or report['captured_optimizer_steps'] != 200:
            raise ValueError('profiler capture incomplete')
    commit = subprocess.check_output(['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip()
    files = [p for directory in ['tools/composition', 'training'] for p in Path(directory).glob('*.py')]
    files += [Path('tools/composition/profile_unsplash_training.sh'), Path('justfile'), Path('DATA_PROVENANCE.md'),
              Path('benchmarks/unsplash_rank_distillation_2026-11-08.md'),
              Path('benchmarks/unsplash_relationships_2026-11-08.md')]
    files += list(Path('docs/experiments').glob('UNSPLASH*.md'))
    files += list(Path('tools/corpus-preview').glob('Cargo.*'))
    files += [p for p in Path('tools/corpus-preview/src').rglob('*') if p.is_file()]
    for root, patterns in [(relationships, ['zensally-relationships*20261108.log']),
                           (roles, []), (rank, ['zensally-unsplash-rank*20261108.log']),
                           (profiles[0], ['zensally-unsplash-rank-training-profile-20261108.log']),
                           (profiles[1], ['zensally-unsplash-rank-training-profile-scoped-20261108.log'])]:
        if (root/'SHA256SUMS.json').exists():
            raise ValueError('packet already frozen: '+str(root))
        # Materialize aliases so a copied packet has no dependency on live paths.
        for path in list(root.rglob('*')):
            if path.is_symlink():
                data = path.read_bytes()
                path.unlink()
                path.write_bytes(data)
        for path in files:
            target = root/'source'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for pattern in patterns:
            for path in Path('/home/lilith/tmp').glob(pattern):
                target = root/'logs'/path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
        identity = dict(build_commit=commit, role_registry_sha256=sha(roles/'registry.json'),
            frozen_test_plan_sha256=sha(rank/'analysis/test-plan.json'), source_roles_changed=False,
            quality_model_promoted=False, human_artistic_quality_measured=False,
            face_coverage_certified=False, paid_model_api_calls=0, diagnostic_maps_saved=False)
        (root/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
        entries = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(root.rglob('*')) if p.is_file()]
        checksum = root/'SHA256SUMS.json'
        checksum.write_text(json.dumps(entries, indent=2)+'\n')
        print(json.dumps(dict(root=str(root), members=len(entries),
            payload_bytes=sum(p['bytes'] for p in entries), manifest_sha256=sha(checksum))), flush=True)


if __name__ == '__main__':
    main()
