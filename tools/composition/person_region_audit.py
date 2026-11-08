#!/usr/bin/env python3
"""Bind unreviewed person-detector rectangles to network-free retention checks."""
import argparse
import json
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    document = json.loads((args.root/'inputs.json').read_text())
    manifest_path = args.root/'object-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['images'] != document['images'] or len(document['images']) != 101:
        raise ValueError('person detector/input identity differs')
    asset = next(m for m in manifest['models'] if m['id'] == 'int8-efficientdet_lite0')
    if sha(Path(asset['path'])) != asset['sha256']:
        raise ValueError('person detector model differs')
    results_path = args.root/'objects-efficientdet/results.json'
    results = json.loads(results_path.read_text())
    if results['model'] != asset['id'] or [r['id'] for r in results['rows']] != [s['id'] for s in document['images']]:
        raise ValueError('complete ordered person detector results required')
    annotations, identities = [], []
    for source, row in zip(document['images'], results['rows']):
        regions = []
        for index, detection in enumerate(row['result']['detections']):
            scores = [c['score'] for c in detection['categories'] if c['name'] == 'person']
            if not scores:
                continue  # Other object classes are outside this person-only audit.
            x, y, width, height = detection['xywh']
            bounds = [max(0, 100*x/source['width']), max(0, 100*y/source['height']),
                      min(100, 100*(x+width)/source['width']), min(100, 100*(y+height)/source['height'])]
            if not (0 <= bounds[0] < bounds[2] <= 100 and 0 <= bounds[1] < bounds[3] <= 100):
                raise ValueError('person detection has no valid visible rectangle')
            regions.append(dict(kind='person_box', description=f'unreviewed detector rectangle {index}',
                                box_percent=bounds, detector_score=max(scores)))
        identities.append(dict(id=source['id'], rgb_sha256=source['source_rgb_sha256']))
        if regions:
            annotations.append(dict(id=source['id'], width=source['native_width'], height=source['native_height'],
                rgb_sha256=source['source_rgb_sha256'], regions=regions))
    args.output.mkdir(parents=True, exist_ok=False)
    annotation = dict(sources=annotations, detector=asset, detector_results_sha256=sha(results_path),
        input_manifest_sha256=sha(manifest_path), training_eligible=False,
        source_population='only detector-positive sources; nondetections are not declared person-free',
        human_rated=False, face_coverage_certified=False)
    identity = dict(cache_sha256=sha(args.root/'runtime-cache/auxiliary.pt'), sources=identities,
                    purpose='runtime source identity binding only; no training admission', training_eligible=False)
    for name, data in [('annotations.json', annotation), ('source-identities.json', identity)]:
        with (args.output/name).open('x') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
    report = dict(screened_sources=101, detector_positive_sources=len(annotations),
                  person_boxes=sum(len(s['regions']) for s in annotations),
                  new_training_labels=False, complete_person_coverage=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
