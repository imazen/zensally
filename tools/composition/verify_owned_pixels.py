#!/usr/bin/env python3
"""Stream packed RGB8 and reconstructed opaque RGBA8 against frozen identities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acquired', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sources = json.loads((args.acquired/'sources.json').read_text())
    checks = []
    for source in sources:
        path = Path(source['directory'])/'source.rgb'
        rgb_hash, rgba_hash = hashlib.sha256(), hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            while raw := stream.read(3*65536):
                size += len(raw)
                rgb_hash.update(raw)
                pixels = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                opaque = np.empty((len(pixels), 4), np.uint8)
                opaque[:, :3] = pixels
                opaque[:, 3] = 255
                rgba_hash.update(opaque)
        valid = (size == source['width']*source['height']*3
                 and rgb_hash.hexdigest() == source['rgb_sha256']
                 and rgba_hash.hexdigest() == source['pixels_sha256'])
        checks.append(dict(id=source['id'], bytes=size, rgb_sha256=rgb_hash.hexdigest(),
            rgba8_sha256=rgba_hash.hexdigest(), manifest_pixels_sha256=source['pixels_sha256'],
            canonical_rgba8_exact=valid))
    report = dict(sources=len(checks), exact_sources=sum(c['canonical_rgba8_exact'] for c in checks),
                  checks=checks, diagnostic_maps_saved=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['sources', 'exact_sources']}))
    if report['exact_sources'] != report['sources']:
        raise ValueError('canonical pixel identities differ; complete source report retained')


if __name__ == '__main__':
    main()
