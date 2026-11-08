#!/usr/bin/env python3
"""Independent PNG readback oracle for owned person-region pixels."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text())
    if len(doc['views']) != 18:
        raise ValueError('registered views differ')
    checks = []
    for row in doc['views']:
        rgb = Path(row['rgb']).read_bytes()
        png = Path(row['png']).read_bytes()
        if (hashlib.sha256(rgb).hexdigest() != row['rgb_sha256']
                or hashlib.sha256(png).hexdigest() != row['png_sha256']):
            raise ValueError('input changed')
        with Image.open(row['png']) as image:
            passed = image.size == (row['width'], row['height']) and image.convert('RGB').tobytes() == rgb
        checks.append(dict(id=row['id'], png_rgb_exact=passed))
    with args.output.open('x') as stream:
        json.dump(dict(checks=checks, training_eligible=False), stream, indent=2)
        stream.write('\n')
    if not all(row['png_rgb_exact'] for row in checks):
        raise ValueError('independent readback differs')
    print(json.dumps(dict(views=len(checks), exact=len(checks))))


if __name__ == '__main__':
    main()
