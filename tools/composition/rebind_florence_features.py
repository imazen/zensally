#!/usr/bin/env python3
"""Reuse pinned larger descriptors with the same cached Florence targets."""
import argparse
import json
from pathlib import Path

import torch

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'focal_cache', 'output']:
        p.add_argument(name, type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    if sha(args.focal_cache) != 'd5b5cc131f7168bf6a981d0fc22f7a664270705d1a7728b832c5a53d3faf5861':
        raise ValueError('registered stronger descriptor cache differs')
    path = args.root/'extent-cache/features.pt'
    current = torch.load(path, map_location='cpu', weights_only=False)
    old = torch.load(args.focal_cache, map_location='cpu', weights_only=False)
    fields = ['id', 'use', 'family_id', 'rgb_sha256']
    if (len(current['sources']) != 56 or len(old['sources']) != 56
            or any(any(a[k] != b[k] for k in fields) for a,b in zip(current['sources'],old['sources']))
            or not torch.equal(current['geometry'],old['geometry'])):
        raise ValueError('descriptor pixel, role, family, order or geometry differs')
    result = dict(current, visual=old['visual'], encoder_sha256=old['encoder_sha256'],
        descriptor_origin=old['descriptor_origin'], parent_extent_cache_sha256=sha(path),
        original_focal_descriptor_cache_sha256=sha(args.focal_cache))
    output = args.output/'extent-cache'
    output.mkdir(parents=True, exist_ok=False)
    torch.save(result, output/'features.pt')
    readback = torch.load(output/'features.pt', map_location='cpu', weights_only=False)
    if (not torch.equal(readback['visual'],old['visual'])
            or any(not torch.equal(readback[k],current[k]) for k in ['geometry','targets','present'])):
        raise ValueError('rebound feature cache readback differs')
    metadata = {k:v for k,v in result.items() if k not in ['visual','geometry','targets','present']}
    metadata['cache_sha256'] = sha(output/'features.pt')
    with (output/'admission.json').open('x') as stream:
        stream.write(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(sources=56, cache_sha256=metadata['cache_sha256'], target_and_geometry_exact=True)))


if __name__ == '__main__':
    main()
