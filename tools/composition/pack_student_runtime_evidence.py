#!/usr/bin/env python3
"""Retain qualified/failed student ports, profiles and bounded review together."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    runtime = args.base/'unsplash-student-runtime'
    original = runtime/'2026-11-08-eval'
    gather = runtime/'2026-11-09-gather-static'
    channels = runtime/'2026-11-09-channel-gather'
    graph = args.base/'graphed-rank-query/2026-11-08'
    review = args.base/'unsplash-rank-review/2026-11-09'
    for root in [original, gather, channels]:
        qualification = json.loads((root/'qualification.json').read_text())
        if any(v['queries'] != 3168 or v['gate_violations'] for v in qualification['runtime_gate'].values()):
            raise ValueError('complete unchanged arithmetic gate required')
        packet = json.loads(gzip.decompress((root/'timing/packet.json.gz').read_bytes()))
        if not packet['complete'] or any(v['status'] != 'completed' for v in packet['invocations']):
            raise ValueError('complete paired timing packet required')
    q = json.loads((graph/'qualification.json').read_text())
    browser = json.loads((review/'browser-unsplash-rank.json').read_text())
    if not q['complete'] or q['scalar_scores_compared'] != 844800 or browser['failures'] or browser['exceptions'] or browser['checks'] != 180:
        raise ValueError('graph query/browser qualification incomplete')
    profile = args.base/'unsplash-rank-training-profile/2026-11-09-graph'
    if (not json.loads((profile/'PROFILE_COMPLETION.json').read_text())['completed'] or
            json.loads((profile/'TARGET_SCOPE.json').read_text())['memory_max_bytes'] != 4294967296):
        raise ValueError('completed bounded target profile required')
    args.output.mkdir(parents=True, exist_ok=False)
    roots = dict(original_invalid=runtime/'2026-11-08', original_qualified=original,
        gather_shape_attempt=runtime/'2026-11-09-gather', gather_qualified=gather,
        channel_gather_qualified=channels,
        batch_gate_gather=runtime/'2026-11-09-batch-qualification-gather',
        batch_gate_channels=runtime/'2026-11-09-batch-qualification',
        graphed_validation=graph, graphed_training_profile=profile,
        cpu_perf=runtime/'2026-11-09-perf', node_profile=runtime/'2026-11-09-node-profile', review=review)
    for name, root in roots.items():
        shutil.copytree(root, args.output/name)  # Materializes aliases.
    # Preserve each producer's actual source revision, including pre-fix attempts.
    producers = dict(original_invalid='52199d3f', original_qualified='7c4a70b5',
        original_timing='fb7a51af', graph_qualification='f519fed5', graph_timing='fa4d75a1',
        graph_training='3fbc46fa', cpu_perf='94d0a54e', node_profile='3d0a5a12',
        gallery='9217bb0b', browser='72e80459', gather_shape_attempt='6c07c26e',
        gather_qualified='ff7d8c32', gather_timing='6f383490')
    producers.update(channel_qualified='cb2aa411', channel_timing='cb2aa411', batch_qualification='d53454a0')
    for root in [roots['batch_gate_gather'], roots['batch_gate_channels']]:
        gate = json.loads((root/'qualification.json').read_text())
        if any(v['requests'] != 1100 or v['gate_violations'] for v in gate['totals'].values()):
            raise ValueError('original batched evaluation gate incomplete or failed')
    for label, commit in producers.items():
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--',
            'tools/composition', 'training', 'tools/crop-first/browser_check.py',
            'tools/crop-first/artifacts.py', 'justfile'], text=True).splitlines()
        for path in paths:
            target = args.output/'producers'/label/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(['git', 'show', commit+':'+path]))
    for name in ['unsplash_student_runtime_2026-11-08.md', 'unsplash_rank_distillation_2026-11-08.md',
                 'unsplash_rank_distillation_2026-11-08.pointer.md']:
        shutil.copy2(Path('benchmarks')/name, args.output/name)
    logs = args.output/'logs'
    logs.mkdir()
    for pattern in ['zensally-unsplash-student*2026110*.log', 'zensally-graphed-rank*20261108.log',
                    'zensally-unsplash-rank-review*20261109.log', 'zensally-unsplash-rank-training-profile-graph*20261109.log']:
        for path in Path('/home/lilith/tmp').glob(pattern):
            shutil.copy2(path, logs/path.name)
    identity = dict(build_commit=subprocess.check_output(
        ['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        producer_commits=producers, selected_student_sha256='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df',
        original_rank_evidence_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        completed_packets=list(roots), original_invalid_reason='export wrapper restored CPU reference training mode',
        gather_shape_attempt_reason='symbolic inferred dimensions rejected before numerical testing',
        graph_training_status='bounded diagnostic capture completed, not a full scientific repeat',
        quality_model_promoted=False, human_artistic_quality_measured=False,
        face_coverage_certified=False, paid_model_api_calls=0, diagnostic_maps_saved=False)
    (args.output/'_MANIFEST.json').write_text(json.dumps(identity, indent=2)+'\n')
    entries = [dict(path=str(p.relative_to(args.output)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(args.output.rglob('*')) if p.is_file()]
    checksum = args.output/'SHA256SUMS.json'
    checksum.write_text(json.dumps(entries, indent=2)+'\n')
    print(json.dumps(dict(root=str(args.output), members=len(entries),
        payload_bytes=sum(v['bytes'] for v in entries), manifest_sha256=sha(checksum))))


if __name__ == '__main__':
    main()
