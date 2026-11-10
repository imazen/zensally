#!/usr/bin/env python3
"""Bind requested detector assets to identical frozen photo and timing inputs."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('screen', type=Path)
    args = parser.parse_args()
    old = json.loads((args.screen/'object-manifest.json').read_text())
    baseline = next(m for m in old['models'] if m['id'] == 'int8-efficientdet_lite0')
    models = [baseline]+json.loads((args.root/'models/models.json').read_text())
    if len(models) != 5:
        raise ValueError('five distinct checkpoint identities required')
    for model in models:
        if sha(Path(model['path'])) != model['sha256']:
            raise ValueError('detector asset differs')
    witnesses = json.loads((args.screen/'crop-proposals/analysis-witnesses.json').read_text())['ids']
    timing_ids = witnesses+['11048', '11214']
    if len(set(timing_ids)) != 8:
        raise ValueError('eight distinct timing photos required')
    images = [dict(s, benchmark=s['id'] in timing_ids) for s in old['images']]
    if len(images) != 101 or sum(s['benchmark'] for s in images) != 8:
        raise ValueError('fixed input cohort differs')
    manifest = dict(images=images, models=models, source_manifest_sha256=sha(args.screen/'object-manifest.json'),
        confidence_threshold=.3, max_results=50, benchmark_ids=timing_ids, face_coverage_certified=False,
        training_eligible=False, paid_model_api_calls=0)
    for name, data in [('manifest.json', manifest), ('scout-manifest.json', dict(manifest,
            images=[s for s in images if s['id'] in ['11048', '11214']]))]:
        with (args.root/name).open('x') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
    print(json.dumps(dict(sources=101, scout_sources=2, timing_sources=8, models=5)))


if __name__ == '__main__':
    main()
