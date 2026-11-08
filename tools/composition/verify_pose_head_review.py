#!/usr/bin/env python3
"""Read owned before/after crop PNGs independently; never create model pixels."""
import argparse
import json
from pathlib import Path
from PIL import Image
from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    doc = json.loads((args.review/'comparison.json').read_text())
    if len(doc['rendered']) != 8 or len(doc['gallery']) != 8 or doc['summary']['requests'] != 44:
        raise ValueError('fixed witness render coverage differs')
    checks = []
    for row in doc['rendered']:
        raw, png = Path(row['rgb']), Path(row['png'])
        if sha(raw) != row['rgb_sha256'] or sha(png) != row['png_sha256']:
            raise ValueError('rendered identity changed')
        with Image.open(png) as image:
            exact = image.size == (row['width'], row['height']) and image.convert('RGB').tobytes() == raw.read_bytes()
        checks.append(dict(file=row['file'], owned_png_rgb_exact=exact))
    with (args.review/'independent-readback.json').open('x') as stream:
        json.dump(dict(checks=checks), stream, indent=2)
        stream.write('\n')
    if not all(c['owned_png_rgb_exact'] for c in checks):
        raise ValueError('independent pixels differ')
    print(json.dumps(dict(rendered_crops=8, exact=8)))


if __name__ == '__main__':
    main()
