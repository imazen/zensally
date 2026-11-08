#!/usr/bin/env python3
"""Screen possible source relationships with the owned scalar fingerprint."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from artifact_identity import sha


def fingerprint(binary, path, width, height, stride):
    result = subprocess.run([str(binary), str(path), str(width), str(height), str(stride)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['previews', 'binary', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    manifest = args.previews/'manifest.json'
    if sha(manifest) != '9d7f1f44c092084391b31a172a8253536686b2b0ec3fca9f5e1250164e4b6cda':
        raise ValueError('registered144-source working pixels differ')
    rows = json.loads(manifest.read_text())['rows']
    if len(rows) != 144 or any(r['training_eligible'] or r['family_role'] is not None for r in rows):
        raise ValueError('registered screening-only role contract differs')
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output/'corpus-fingerprint'
    shutil.copy2(args.binary, binary)
    protocol = dict(preview_manifest_sha256=sha(manifest), binary_sha256=sha(binary),
        era='zensally-working-srgb1280-native-linear-owner-v1', threshold=10,
        expected_sources=144, expected_pairs=10296, calibration_fixture_budget=16*1024**2,
        owner_commit='ed956176e02e335d4a6efb3c31724e2aaf361d34',
        roles_assigned=False, training_labels_admitted=False, diagnostic_maps_saved=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    first = rows[0]
    packed = Path(first['rgb']).read_bytes()
    width, height = first['width'], first['height']
    if sha(Path(first['rgb'])) != first['rgb_sha256']:
        raise ValueError('calibration source differs')
    stride = width*3+17
    padded = bytearray([219])*(stride*height)
    for y in range(height):
        padded[y*stride:y*stride+width*3] = packed[y*width*3:(y+1)*width*3]
    full, short = args.output/'calibration-padded.rgb', args.output/'calibration-short-final.rgb'
    full.write_bytes(padded)
    short.write_bytes(padded[:(height-1)*stride+width*3])
    if full.stat().st_size+short.stat().st_size > protocol['calibration_fixture_budget']:
        raise ValueError('bounded calibration fixture budget exceeded')
    values = [fingerprint(binary, Path(first['rgb']), width, height, width*3)]
    values += [fingerprint(binary, path, width, height, stride) for path in [full, short]]
    if values[0] != values[1] or values[0] != values[2]:
        raise ValueError('identity/stride calibration differs')
    (args.output/'calibration.json').write_text(json.dumps(dict(source_id=first['id'], values=values,
        input_rgb_sha256=first['rgb_sha256'], padded_sha256=sha(full), short_final_sha256=sha(short),
        exact=True), indent=2)+'\n')
    hashes = []
    for row in rows:
        path = Path(row['rgb'])
        if sha(path) != row['rgb_sha256']:
            raise ValueError('immutable working pixels differ')
        value = fingerprint(binary, path, row['width'], row['height'], row['width']*3)
        hashes.append(dict(id=row['id'], **value, rgb_sha256=row['rgb_sha256'],
            original_sha256=row['source']['sha256'], photographer_group=row['source']['provisional_group'],
            source_url=row['source']['source_url']))
        (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), completed=len(hashes),
            required=144, checked_unix=time.time()))+'\n')
    (args.output/'fingerprints.json').write_text(json.dumps(hashes, indent=2)+'\n')
    parents = {r['id']: r['id'] for r in hashes}
    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier
    edges, pairs = [], []
    for left, right in combinations(hashes, 2):
        distance = (int(left['hash'], 16)^int(right['hash'], 16)).bit_count()
        reasons = []
        if left['photographer_group'] == right['photographer_group']:
            reasons.append('shared_provisional_photographer')
        if left['original_sha256'] == right['original_sha256']:
            reasons.append('exact_original_sha256')
        if distance <= 10:
            reasons.append('possible_whole_image_fingerprint_relationship')
        pair = dict(left=left['id'], right=right['id'], hamming_distance=distance,
                    same_photographer=left['photographer_group'] == right['photographer_group'])
        pairs.append(pair)
        if reasons:
            edges.append(dict(pair, reasons=reasons))
            a, b = sorted([find(left['id']), find(right['id'])])
            parents[b] = a
    components = defaultdict(list)
    for row in hashes:
        components[find(row['id'])].append(row['id'])
    if len(pairs) != 10296:
        raise ValueError('complete unordered pair screen required')
    (args.output/'pair-distances.json').write_text(json.dumps(pairs, indent=2)+'\n')
    (args.output/'relationship-edges.json').write_text(json.dumps(edges, indent=2)+'\n')
    (args.output/'components.json').write_text(json.dumps(sorted(components.values()), indent=2)+'\n')
    summary = dict(sources=144, pairs=len(pairs), provisional_photographer_groups=len({r['photographer_group'] for r in hashes}),
        fingerprint_flags=sum(p['hamming_distance'] <= 10 for p in pairs),
        across_photographer_flags=sum(p['hamming_distance'] <= 10 and not p['same_photographer'] for p in pairs),
        conservative_components=len(components), component_size_histogram=dict(Counter(map(len, components.values()))),
        calibration_exact=True, complete_perceptual_duplicate_recall_certified=False,
        roles_assigned=False, training_labels_admitted=False, face_coverage_certified=False,
        canonical_holdout_pixels_opened=False, paid_model_api_calls=0)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete', completed=144,
        required=144, checked_unix=time.time()))+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
