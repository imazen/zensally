#!/usr/bin/env python3
"""Describe fixed-cohort detector counts/agreement without inventing recall."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from artifact_identity import sha


def people(row):
    return [d for d in row['result']['detections'] if any(c['name'] == 'person' for c in d['categories'])]


def iou(a, b):
    x, y, w, h = a['xywh']
    u, v, p, q = b['xywh']
    overlap = max(0, min(x+w, u+p)-max(x, u))*max(0, min(y+h, v+q)-max(y, v))
    return overlap/(w*h+p*q-overlap) if w*h+p*q > overlap else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('previous_screen', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root/'manifest.json').read_text())
    models, results, positives = [], {}, {}
    ids = [r['id'] for r in manifest['images']]
    for model in manifest['models']:
        path = args.root/'full'/model['id']/'results.json'
        rows = json.loads(path.read_text())['rows']
        if [r['id'] for r in rows] != ids:
            raise ValueError('complete101source input ordering required')
        result = {r['id']: people(r) for r in rows}
        results[model['id']] = result
        positives[model['id']] = {key for key, value in result.items() if value}
        models.append(dict(id=model['id'], bytes=model['bytes'], sha256=model['sha256'],
            sources_with_person=len(positives[model['id']]), person_boxes=sum(map(len, result.values())),
            results_sha256=sha(path)))
    baseline_path = args.root/'full/int8-efficientdet_lite0/results.json'
    baseline = json.loads(baseline_path.read_text())
    previous = json.loads((args.previous_screen/'objects-efficientdet/results.json').read_text())
    agreement = []
    for left, right in combinations(results, 2):
        matched = 0
        for source in ids:
            l, r = results[left][source], results[right][source]
            choices = sorted(((iou(a, b), i, j) for i, a in enumerate(l) for j, b in enumerate(r)), reverse=True)
            used_left, used_right = set(), set()
            for overlap, i, j in choices:
                if overlap < .5:
                    break
                if i not in used_left and j not in used_right:
                    matched += 1
                    used_left.add(i)
                    used_right.add(j)
        agreement.append(dict(left=left, right=right, matched_boxes_at_iou_half=matched,
            shared_positive_sources=len(positives[left] & positives[right]),
            left_only_positive_sources=sorted(positives[left]-positives[right]),
            right_only_positive_sources=sorted(positives[right]-positives[left])))
    witnesses = json.loads((args.previous_screen/'witness-observations.json').read_text())['sources']
    report = dict(sources=101, models=models, pairwise=agreement,
        original_lite0_rows_exact=baseline['rows'] == previous['rows'],
        witnesses=[dict(id=w['id'], prior_observation=w['observation'],
            person_counts={m:len(rows[w['id']]) for m, rows in results.items()}) for w in witnesses],
        measured_human_precision_recall=False, face_coverage_certified=False,
        training_labels_admitted=False, paid_model_api_calls=0)
    with (args.root/'coverage-analysis.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(models=models, original_lite0_rows_exact=report['original_lite0_rows_exact'],
                         witnesses=report['witnesses'])))


if __name__ == '__main__':
    main()
