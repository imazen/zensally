#!/usr/bin/env python3
"""Freeze paired human masks before running preprocessing experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False, parents=True)
images = sorted((a.dataset / 'DUTS-TE-Image').glob('*.jpg'))
assert len(images) == 5019, f'Expected 5019 complete DUTS-TE images, got {len(images)}'
rows = []
for image in images:
    ident = image.stem
    mask = a.dataset / 'DUTS-TE-Mask' / f'{ident}.png'
    rank = hashlib.sha256(ident.encode()).hexdigest()
    rows.append([ident, 'calibration' if int(rank[:2], 16) % 4 == 0 else 'evaluation',
                 image.relative_to(a.dataset), mask.relative_to(a.dataset),
                 hashlib.sha256(image.read_bytes()).hexdigest(), hashlib.sha256(mask.read_bytes()).hexdigest(), rank])
for split in ['calibration', 'evaluation']:
    stress = {r[0] for r in sorted((r for r in rows if r[1] == split), key=lambda r: r[6])[:64]}
    for r in rows:
        if r[1] == split:
            r.append(str(r[0] in stress).lower())
with (a.output / 'inputs.tsv').open('x') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['id', 'split', 'image', 'mask', 'image_sha256', 'mask_sha256', 'rank', 'stress'])
    w.writerows(rows)
meta = {'dataset': 'DUTS-TE', 'url': 'https://saliencydetection.net/duts/',
        'source_archive_sha256': 'd9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104',
        'manifest_sha256': hashlib.sha256((a.output / 'inputs.tsv').read_bytes()).hexdigest(),
        'images': len(rows), 'split_counts': {s: sum(r[1] == s for r in rows) for s in ['calibration', 'evaluation']},
        'stress_per_split': 64, 'model_training': 'DUTS-TR', 'role': 'evaluation-only; no training or model-weight changes'}
(a.output / 'source.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta))
