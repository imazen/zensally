#!/usr/bin/env python3
"""Bind/describe a second unlabeled detector cohort without admitting labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_efficientdet_family import people
from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    for name in ['previews', 'family', 'output']:
        prepare.add_argument(name, type=Path)
    analyze = sub.add_parser('analyze')
    analyze.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        views = json.loads((args.previews/'manifest.json').read_text())
        original = json.loads((args.family/'manifest.json').read_text())
        if len(views['rows']) != 144 or len({r['id'] for r in views['rows']}) != 144:
            raise ValueError('registered144-source views differ')
        models = []
        for model in original['models']:
            path = args.family/'models'/Path(model['path']).name
            if sha(path) != model['sha256']:
                raise ValueError('frozen detector identity differs')
            models.append(dict(model, path=str(path)))
        images = []
        for row in views['rows']:
            if row['training_eligible'] or row['family_role'] is not None or not row['independent_png_rgb_exact']:
                raise ValueError('registered screening-only contract differs')
            if sha(Path(row['rgb'])) != row['rgb_sha256'] or sha(Path(row['png'])) != row['png_sha256']:
                raise ValueError('owned working pixels differ')
            images.append(dict(id=row['id'], width=row['width'], height=row['height'], rgb=row['rgb'],
                rgb_sha256=row['rgb_sha256'], benchmark=False, native_size=row['native_size'],
                preview_png_name=Path(row['png']).name, source=row['source'], synthetic=False))
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(images=images, models=models, source_manifest_sha256=sha(args.previews/'manifest.json'),
            weight_packet_sha256=sha(args.family/'SHA256SUMS.json'), confidence_threshold=.3, max_results=50,
            preprocessing=views['protocol']['preprocessing_version'], training_eligible=False,
            face_coverage_certified=False, roles_assigned=False, timing_measured=False, paid_model_api_calls=0)
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(sources=144, checkpoints=len(models))))
        return
    manifest = json.loads((args.root/'manifest.json').read_text())
    ids = [r['id'] for r in manifest['images']]
    models, counts = [], {}
    for model in manifest['models']:
        path = args.root/'full'/model['id']/'results.json'
        rows = json.loads(path.read_text())['rows']
        if [r['id'] for r in rows] != ids:
            raise ValueError('complete ordered144-source results required')
        count = {r['id']: len(people(r)) for r in rows}
        counts[model['id']] = count
        classes = Counter(c['name'] for r in rows for d in r['result']['detections'] for c in d['categories'])
        models.append(dict(id=model['id'], bytes=model['bytes'], person_boxes=sum(count.values()),
            sources_with_person=sum(v > 0 for v in count.values()), all_class_instances=dict(classes),
            results_sha256=sha(path)))
    disagreement = {i: max(c[i] for c in counts.values())-min(c[i] for c in counts.values()) for i in ids}
    maximum = {i: max(c[i] for c in counts.values()) for i in ids}
    order = [min(ids, key=lambda i: (-disagreement[i], i)), min(ids, key=lambda i: (-maximum[i], i))]
    order += sorted(i for i in ids if counts['int8-efficientdet_lite0'][i] == 0 and maximum[i] > 0)
    order += sorted(ids)
    witnesses = list(dict.fromkeys(order))[:4]
    report = dict(sources=144, models=models, posthoc_visual_witness_ids=witnesses,
        source_counts={i: {m: c[i] for m, c in counts.items()} for i in ids},
        human_precision_recall_measured=False, face_coverage_certified=False,
        training_labels_admitted=False, roles_assigned=False, paid_model_api_calls=0)
    with (args.root/'coverage-analysis.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(models=models, posthoc_visual_witness_ids=witnesses)))


if __name__ == '__main__':
    main()
