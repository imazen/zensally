#!/usr/bin/env python3
"""Network-free arbitrary-crop region retention; never invent artistic grades."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_fraction(crop, region):
    for box in [crop, region]:
        if len(box) != 4 or any(not math.isfinite(x) for x in box) or not (
                box[0] < box[2] and box[1] < box[3]):
            raise ValueError('finite positive crop and region bounds required')
    width = max(0., min(crop[2], region[2]) - max(crop[0], region[0]))
    height = max(0., min(crop[3], region[3]) - max(crop[1], region[1]))
    return width * height / ((region[2] - region[0]) * (region[3] - region[1]))


def evaluate(annotation, bank):
    if (annotation['id'], annotation['width'], annotation['height']) != (bank['id'], bank['width'], bank['height']):
        raise ValueError('annotation/bank source geometry differs')
    width, height = bank['width'], bank['height']
    bounds = [0, 0, width, height]
    regions = []
    for region in annotation['regions']:
        p = region['box_percent']
        box = [math.floor(p[0]*width/100), math.floor(p[1]*height/100),
               math.ceil(p[2]*width/100), math.ceil(p[3]*height/100)]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('protected region is not contained in source')
        regions.append(dict(region, pixel_box=box))
    rows = []
    for task in bank['tasks']:
        for selection in task['selected']:
            index = selection['detected_head_guarded_candidate']
            action = 'contain' if index is None else 'crop'
            box = bounds
            if index is not None:
                candidate = task['candidates'][index]
                if 'pixel_box' in candidate:
                    box = candidate['pixel_box']
                else:
                    b = candidate['box']
                    box = [b[0]*width, b[1]*height, b[2]*width, b[3]*height]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    raise ValueError('crop extends outside source')
            scores = [dict(kind=r['kind'], description=r['description'],
                           retained_box_area=retained_fraction(box, r['pixel_box']),
                           fully_contained=box[0] <= r['pixel_box'][0] and box[1] <= r['pixel_box'][1]
                           and box[2] >= r['pixel_box'][2] and box[3] >= r['pixel_box'][3]) for r in regions]
            rows.append(dict(id=bank['id'], ratio=task['ratio'], mode=selection['mode'],
                action=action, crop_pixels=box, regions=scores,
                all_regions_contained=all(r['fully_contained'] for r in scores),
                face_coverage_certified=False, aesthetic_grade=None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--admission', required=True, type=Path)
    parser.add_argument('--scope', default='eight approximate unreviewed whole-product/story witness lists; rectangle-area retention is not pixel segmentation, artistic quality, precision or face recall')
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text())
    admission = json.loads(args.admission.read_text())
    lookup = {s['id']:s for s in admission['sources']}
    bank_summary = json.loads((args.bank/'summary.json').read_text())
    # Older semantic witnesses bind source hashes in their annotation file;
    # full model banks additionally bind the exact packed source cache.
    if 'source_cache_sha256' in bank_summary and bank_summary['source_cache_sha256'] != admission['cache_sha256']:
        raise ValueError('bank/admission source cache differs')
    rows, identities = [], []
    for source in annotations['sources']:
        if lookup[source['id']]['rgb_sha256'] != source['rgb_sha256']:
            raise ValueError('annotation/admission source pixels differ')
        file = args.bank/(source['id']+'.json')
        bank = json.loads(file.read_text())
        rows.extend(evaluate(source, bank))
        identities.append(dict(id=source['id'], annotation_rgb_sha256=source['rgb_sha256'],
                               bank_file=str(file), bank_sha256=digest(file)))
    counts, kinds, ratios = Counter(), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        for counts_for in [counts, ratios[row['ratio']]]:
            counts_for['requests'] += 1
            counts_for[row['action']] += 1
            counts_for['offered_crop_loses_region'] += int(row['action']=='crop' and not row['all_regions_contained'])
        if row['action']=='crop':
            for region in row['regions']:
                kinds[region['kind']]['offered_region_instances'] += 1
                kinds[region['kind']]['clipped'] += int(not region['fully_contained'])
    result = dict(summary=dict(counts), by_kind={k:dict(v) for k,v in kinds.items()},
        by_ratio={k:dict(v) for k,v in ratios.items()}, requests=rows,
        annotation_sha256=digest(args.annotations), bank_identities=identities,
        source_admission_sha256=digest(args.admission), source_cache_sha256=admission['cache_sha256'],
        scope=args.scope,
        cached_aesthetic_judgments_extrapolated=False, diagnostic_maps_saved=False,
        new_labels_admitted=False, face_coverage_certified=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ['summary','by_kind','by_ratio']}, indent=2))


if __name__ == '__main__':
    main()
