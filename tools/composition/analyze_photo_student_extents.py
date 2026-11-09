#!/usr/bin/env python3
"""Compare completed photographic feature transfer to unchanged geometry controls."""
import argparse
import json
from pathlib import Path

from analyze_florence_extents import study
from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('original',type=Path)
    args = p.parse_args()
    result = study(args.root/'extent-fits')
    for seed in [31,137,911]:
        name = 'aspect-seed'+str(seed)
        for file in ['history.json','best-predictions.json']:
            if json.loads((args.root/'extent-fits'/name/file).read_text()) != json.loads(
                    (args.original/'extent-fits'/name/file).read_text()):
                raise ValueError('matched geometry-only control changed')
    result.update(matched_aspect_controls_exact=True,cache_sha256=sha(args.root/'extent-cache/features.pt'),
        encoder_sha256='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df',
        parent_target_cache_sha256='f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f',
        scope='eight reused synthetic development families; frozen photographic encoder; no artistic-quality or complete-face claim',
        model_promoted=False,paid_model_api_calls=0,new_test_inference=False)
    with (args.root/'analysis.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
