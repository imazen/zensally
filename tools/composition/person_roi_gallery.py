#!/usr/bin/env python3
"""Show actual person ROI PNGs and detector/pose coordinates for inspection."""
import argparse
import json
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--asset-base', required=True)
    args = parser.parse_args()
    manifest = args.root/'person-roi-inputs/manifest.json'
    doc = json.loads(manifest.read_text())
    oracle = json.loads((args.root/'person-roi-input-oracle.json').read_text())
    if len(oracle['checks']) != 18 or not all(r['png_rgb_exact'] for r in oracle['checks']):
        raise ValueError('independent readback required')
    results = {}
    for model in doc['models']:
        data = json.loads((args.root/'person-roi-results'/(model['id']+'.json')).read_text())
        if data['manifest_sha256'] != sha(manifest):
            raise ValueError('result/input identity differs')
        results[model['id']] = {r['id']: r['result'] for r in data['rows']}
    rows = []
    for row in doc['views']:
        if sha(Path(row['png'])) != row['png_sha256']:
            raise ValueError('owned PNG changed')
        rows.append(dict(id=row['id'], width=row['width'], height=row['height'],
            image=args.asset_base.rstrip('/')+'/'+Path(row['png']).name,
            results={key: value[row['id']] for key, value in results.items()}))
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('person_roi_gallery.html').read_text()
    (args.output/'index.html').write_text(template.replace('/*DOCUMENT*/null', json.dumps(rows).replace('<', '\\u003c')))
    print(json.dumps(dict(views=len(rows), new_pixels_generated=False)))


if __name__ == '__main__':
    main()
