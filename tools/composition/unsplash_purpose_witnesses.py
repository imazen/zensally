#!/usr/bin/env python3
"""Source-bound unreviewed purpose regions for the fixed six-image review."""
import argparse
import json
from pathlib import Path

from artifact_identity import sha


# Full sources were inspected; annotations remain unblinded model estimates.
# Each tuple is a request-specific nomination, not a generic purpose label.
SCENES = {
    'v5C47rEShZ4': dict(heads=[], requests=dict(
        catalogue=('complete phone', [45.5, 28, 63.5, 67]),
        hero=('phone display and holding fingers', [48, 35, 64, 64]),
        detail=('phone home button and lower frame', [53, 57, 62, 67]))),
    'kRnM-UJpK6I': dict(heads=[[45, 41.5, 58.5, 52]], requests=dict(
        catalogue=('complete featured book', [32, 49.5, 49.5, 60]),
        hero=('reader and book context', [32, 42, 63, 61]),
        detail=('nominated book-cover feature', [38, 53, 49, 60]))),
    'MqOGVcLcJ3E': dict(heads=[], requests=dict(
        catalogue=('complete wallet', [31.5, 39.5, 72, 76.5]),
        hero=('wallet front and embossed identity', [34, 53, 70, 76.5]),
        detail=('embossed wallet brand', [35.5, 60, 43.5, 72]))),
    'nAMkUTfLjqI': dict(heads=[[46, 46, 54, 53]], requests=dict(
        hero=('central arch and passage', [34, 36, 61, 58]),
        detail=('hanging plants, while retaining supplied heads', [26, 18, 58, 39]))),
    '04dgurD3dn8': dict(heads=[], requests=dict(
        hero=('central tree fork and canopy', [23, 20, 82, 63]),
        detail=('moss-covered branch detail', [29, 30, 59, 46]))),
    'FFPu5QDn40Y': dict(heads=[], requests=dict(
        hero=('boat and adjacent water', [51, 56, 83, 69]),
        detail=('whole nominated moon', [16, 13, 31, 25]))),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('registered source roles differ')
    sources = {s['source']['provider_id']:s for s in json.loads(args.registry.read_text())['sources']}
    rows = []
    for key, scene in SCENES.items():
        source = sources[key]
        if source['role'] != 'dev':
            raise ValueError('development-only witnesses required')
        regions = scene['heads']+[box for _,box in scene['requests'].values()]
        if any(not (0 <= l < r <= 100 and 0 <= t < b <= 100) for l,t,r,b in regions):
            raise ValueError('invalid percentage annotation')
        rows.append(dict(id=source['id'], family_id=source['family_id'], role='dev',
            native_size=source['native_size'], png=source['png'], png_sha256=source['png_sha256'],
            rgb_sha256=source['rgb_sha256'], source_url=source['source']['source_url'],
            photographer=source['source']['photographer'], detector_person_boxes=source['detected_person_boxes'],
            head_candidate_boxes_percent=scene['heads'],
            requests={purpose:dict(description=description, required_box_percent=box)
                      for purpose,(description,box) in scene['requests'].items()},
            human_rated=False, face_coverage_certified=False, training_eligible=False))
    doc = dict(schema='zensally-purpose-witness-v1', sources=rows, registry_sha256=sha(args.registry),
        annotation_origin='unblinded Codex visual estimates after source and baseline inspection',
        model_has_purpose_input=False, training_updated=False, model_reselected=False,
        new_test_inference=False, paid_model_api_calls=0, expected_requests=330)
    with args.output.open('x') as stream:
        stream.write(json.dumps(doc, indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows), nominations=sum(len(s['requests']) for s in rows), requests=330)))


if __name__ == '__main__':
    main()
