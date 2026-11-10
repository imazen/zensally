#!/usr/bin/env python3
"""Audit cached local region geometry; compare three unreviewed product marks."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_efficientdet_family import iou
from artifact_identity import sha
from region_retention import retained_fraction

GROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'
DENSE = '<DENSE_REGION_CAPTION>'
CAPTION = '<MORE_DETAILED_CAPTION>'
NOUNS = {'unsplash:MqOGVcLcJ3E':'wallet', 'unsplash:v5C47rEShZ4':'phone', 'unsplash:kRnM-UJpK6I':'book'}


def xywh(box):
    return dict(xywh=[box[0],box[1],box[2]-box[0],box[3]-box[1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['scout', 'purposes', 'registry', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.scout/'scout-summary.json').read_text())
    purpose = json.loads((args.purposes/'comparison.json').read_text())
    if (summary['completed'] != 18 or summary['invalid_calls'] or
            sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6'):
        raise ValueError('complete registered scout and source registry required')
    registry = {s['id']:s for s in json.loads(args.registry.read_text())['sources']}
    purpose_rows = {s['id']:s for s in purpose['sources']}
    grouped, counts, invalid = {}, Counter(), []
    for call in summary['calls']:
        path = args.scout/'responses'/call['response']
        response = json.loads(path.read_text())
        contract = response['contract']
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        source = registry[call['id']]
        key = source['source']['provider_id']
        input_meta = json.loads((args.scout/'inputs'/(key+'.json')).read_text())
        if (digest != path.stem or digest != response['contract_sha256'] or contract['source_id'] != call['id']
                or contract['source_role'] != 'dev' or source['role'] != 'dev'
                or contract['native_size'] != source['native_size']
                or input_meta['contract']['source_rgb_sha256'] != source['rgb_sha256']
                or sha(args.scout/'inputs'/(key+'.rgb')) != contract['owned_input_sha256']
                or response['postprocess_error'] or response['token_cap_reached'] or not response['ended_with_eos']):
            raise ValueError('source-bound cached inference receipt differs or is incomplete')
        entry = grouped.setdefault(call['id'], dict(id=call['id'], family_id=source['family_id'], role='dev',
            native_size=source['native_size'], png=source['png'], png_sha256=source['png_sha256'],
            source_url=source['source']['source_url'], photographer=source['source']['photographer'], tasks={}))
        if call['task'] in entry['tasks']:
            raise ValueError('duplicate source/task result')
        task = dict(response_file=call['response'], response_sha256=sha(path), prompt=contract['prompt'],
                    parsed=response['parsed'], raw_text=response['generated_text'], boxes=[])
        if call['task'] != CAPTION:
            value = response['parsed'][call['task']]
            boxes, labels = value['bboxes'], value['labels']
            if len(boxes) != len(labels):
                raise ValueError('region/label counts differ')
            width, height = source['native_size']
            for box,label in zip(boxes,labels):
                valid = (len(box) == 4 and all(math.isfinite(v) for v in box)
                    and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                item = dict(box=box, label=label, valid_geometry=valid)
                task['boxes'].append(item)
                counts['boxes'] += 1
                counts['valid_boxes'] += int(valid)
                counts[call['task']+'_boxes'] += 1
                if not valid:
                    invalid.append(dict(id=call['id'],task=call['task'],**item))
        entry['tasks'][call['task']] = task
    if len(grouped) != 6 or any(set(s['tasks']) != {CAPTION,DENSE,GROUNDING} for s in grouped.values()):
        raise ValueError('all six sources and three tasks required')
    product_rows = []
    for source_id,noun in NOUNS.items():
        source = grouped[source_id]
        reference = next(t['focus_pixels'] for t in purpose_rows[source_id]['tasks'] if t['purpose']=='catalogue')
        source['approximate_catalogue_region'] = reference
        row = dict(id=source_id, noun=noun, approximate_catalogue_region=reference, methods={})
        for task in [GROUNDING,DENSE]:
            matches = []
            for item in source['tasks'][task]['boxes']:
                if noun in item['label'].lower() and item['valid_geometry']:
                    matches.append(dict(**item, iou_with_approximate_region=iou(xywh(item['box']), xywh(reference)),
                        approximate_region_area_retained=retained_fraction(item['box'], reference)))
            row['methods'][task] = dict(matches=matches, matching_labels=len(matches),
                first_match_iou=None if not matches else matches[0]['iou_with_approximate_region'],
                oracle_best_match_iou=None if not matches else max(m['iou_with_approximate_region'] for m in matches))
        product_rows.append(row)
    means = {}
    for task in [GROUNDING,DENSE]:
        values = [r['methods'][task]['first_match_iou'] for r in product_rows]
        if any(v is None for v in values):
            means[task] = dict(supported=sum(v is not None for v in values), first_match_mean_iou=None)
        else:
            means[task] = dict(supported=3, first_match_mean_iou=statistics.mean(values),
                oracle_best_match_mean_iou=statistics.mean(r['methods'][task]['oracle_best_match_iou'] for r in product_rows))
    result = dict(sources=list(grouped.values()), counts=dict(counts), invalid_boxes=invalid,
        product_comparisons=product_rows, product_means=means,
        scope='agreement with three approximate unreviewed product marks; not true IoU, recall, completeness or artistic quality',
        label_selection='substring noun matching; first match deterministic; oracle best diagnostic is not a deployable selector',
        teacher_suitable_for_training=False, training_labels_admitted=False,
        human_artistic_quality_measured=False, face_coverage_certified=False,
        source_roles_changed=False, new_test_inference=False, paid_model_api_calls=0)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'], product_means=means,
        product_first_iou={r['id']:{k:v['first_match_iou'] for k,v in r['methods'].items()} for r in product_rows})))
    if invalid:
        raise ValueError('scout emitted invalid native region geometry; retained every result')


if __name__ == '__main__':
    main()
