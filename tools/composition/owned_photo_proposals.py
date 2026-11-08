#!/usr/bin/env python3
"""Unadmitted photographic proposals with cached image analysis and integer bounds."""
import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from artifact_identity import sha
from attention import heads as detected_heads
from cached_focalnet import CachedContext, prepared
from constrained_bank import candidates
from focalnet_bank import RATIOS
from owned_photo_plan import analysis_witnesses
from query_selection import eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'models', 'split', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--additional-heads', type=Path)
    args = parser.parse_args()
    if bool(args.ids) != bool(args.additional_heads):
        raise ValueError('head follow-up requires both IDs and additional-region identity')
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.models/'focalnet-source/src'))
    from focalnet.imaging import Letterbox, restore_map
    from focalnet.crop import Crop, score_crop
    document = json.loads((args.root/'inputs.json').read_text())
    images = {s['id']: s for s in document['images']}
    cache = args.root/'runtime-cache/auxiliary.pt'
    auxiliary = torch.load(cache, map_location='cpu', weights_only=False)
    if len(auxiliary['sources']) != 101 or auxiliary['training_eligible']:
        raise ValueError('registered unadmitted runtime cohort required')
    additions = {}
    if args.additional_heads:
        doc = json.loads(args.additional_heads.read_text())
        additions = {s['id']: s for s in doc['sources']}
        if (set(args.ids) != set(additions) or len(args.ids) != len(set(args.ids))
                or not set(args.ids) <= set(images) or len(args.ids) > 8
                or doc['face_coverage_certified'] or doc['training_eligible']):
            raise ValueError('research-only region/source contract differs')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'analysis-witnesses.json').open('x') as stream:
        json.dump(dict(ids=args.ids or analysis_witnesses(auxiliary['sources']),
            selection='explicit posthoc region witnesses' if args.ids else
            'fixed metadata/category hash order with distinct source families; before numerical errors'), stream, indent=2)
    context = CachedContext(args.split, threads=4, query_threads=1)
    counts = Counter()
    for i, source in enumerate(auxiliary['sources']):
        if args.ids and source['id'] not in additions:
            continue
        frame = images[source['id']]
        face_path = args.root/'geometry'/(source['id']+'-faces.json')
        pose_path = args.root/'geometry'/(source['id']+'-pose.json')
        faces = json.loads(face_path.read_text())
        pose = json.loads(pose_path.read_text())
        if any(d['id'] != source['id'] or d['role'] != 'screening'
               or d['rgb_sha256'] != frame['rgb_sha256'] for d in [faces, pose]):
            raise ValueError('geometry/source identity differs')
        width, height = source['width'], source['height']
        detections = faces['result']['detections']
        if len(detections) > 256:
            raise ValueError('face count exceeds this bounded screening tool; no detection discarded')
        # Reuse the existing face-to-head padding in bounded chunks. This is
        # geometry only: no gaze network, label or coverage certification.
        regions = []
        for start in range(0, len(detections), 16):
            regions += detected_heads(frame, dict(result=dict(detections=detections[start:start+16])))
        boxes = [r['head'] for r in regions]
        protected = [[math.floor(b[0]*width), math.floor(b[1]*height),
                      math.ceil(b[2]*width), math.ceil(b[3]*height)] for b in boxes]
        if source['id'] in additions:
            addition = additions[source['id']]
            if sha(args.root/'crop-proposals'/(source['id']+'.json')) != addition['bank_sha256']:
                raise ValueError('additional heads refer to a different original bank')
            for region in addition['added_regions']:
                l, t, r, b = region['pixel_box']
                if not (all(isinstance(v, int) for v in [l, t, r, b])
                        and 0 <= l < r <= width and 0 <= t < b <= height):
                    raise ValueError('invalid source-pixel head region')
                protected.append([l, t, r, b])
                boxes.append([l/width, t/height, r/width, b/height])
        image, content = prepared(auxiliary, i)
        token = context.analyze(image, content, (source['id'], source['rgb_sha256']))
        x0, y0, x1, y1 = np.rint(content[0]*256).astype(int)
        importance = restore_map(context.outputs['importance'][0, 0],
                                 Letterbox(256, x0, y0, x1-x0, y1-y0))
        tasks = []
        for ratio, _ in RATIOS:
            p, q = map(int, ratio.split(':'))
            integer_boxes, maximum_w, maximum_h = candidates(width, height, (p, q), protected)
            padded = np.zeros((1, 128, 4), np.float32)
            if integer_boxes:
                padded[0, :len(integer_boxes)] = np.array(integer_boxes, np.float64)/[width, height, width, height]
            scores = context.score(padded, token)
            values = []
            for j, (l, t, r, b) in enumerate(integer_boxes):
                if ((r-l)*q != (b-t)*p or not (0 <= l < r <= width and 0 <= t < b <= height)
                        or any(l > h[0] or t > h[1] or r < h[2] or b < h[3] for h in protected)):
                    raise ValueError('independent integer crop/region guard failed')
                values.append(dict(box=[l/width, t/height, r/width, b/height], pixel_box=[l, t, r, b],
                    score=float(scores[0, j]), contains_detected_heads=True,
                    retained_importance=float(score_crop(importance, Crop(l, t, r-l, b-t, 0), width, height))))
            task = dict(ratio=ratio, candidates=values, selected=[])
            for mode in ['minimal', 'maximal']:
                mask = eligible(task, width, height, mode)
                indices = np.flatnonzero(mask)
                selected = int(indices[np.argmax(scores[0, indices])]) if len(indices) else None
                aggressive = any(b[2]-b[0] < maximum_w for b in integer_boxes)
                fallback = mode == 'maximal' and bool(integer_boxes) and not aggressive
                task['selected'].append(dict(mode=mode, raw_candidate=selected,
                    detected_head_guarded_candidate=selected, minimal_fallback=fallback,
                    guarded_action='contain' if selected is None else 'research_crop',
                    face_coverage_certified=False, deployable=False))
                counts['requests'] += 1
                counts['contain' if selected is None else 'research_crop'] += 1
                counts['least_crop_fallback'] += int(fallback)
            tasks.append(task)
        row = dict(id=source['id'], family_id=source['family_id'], use='screening',
            width=width, height=height, heads=boxes, protected_head_pixels=protected,
            tasks=tasks, pose=pose['result']['landmarks'], license=source['license'],
            canonical_url=source['canonical_url'], provenance=source['provenance'],
            face_geometry_sha256=sha(face_path), pose_geometry_sha256=sha(pose_path),
            diagnostic_maps_saved=False, face_coverage_certified=False, deployable=False,
            new_training_labels_admitted=False)
        with (args.output/(source['id']+'.json')).open('x') as stream:
            json.dump(row, stream)
            stream.write('\n')
        counts['sources'] += 1
        counts['sources_with_faces'] += int(bool(detections))
        counts['detected_faces'] += len(detections)
        counts['sources_with_pose'] += int(bool(pose['result']['landmarks']))
        counts['detected_poses'] += len(pose['result']['landmarks'])
        print(json.dumps(dict(completed=counts['sources'], id=source['id'])), flush=True)
    report = dict(counts, source_cache_sha256=sha(cache), split_identity_sha256=sha(args.split/'identity.json'),
        contract='existing exact-integer candidate bounds, retained importance tolerance .05 then frozen rank; supplied detector regions only',
        analysis_threads=4, query_threads=1, new_labels_admitted=False,
        face_coverage_certified=False, quality_established=False, diagnostic_maps_saved=False)
    if args.additional_heads:
        report['additional_heads_sha256'] = sha(args.additional_heads)
        report['contract'] += '; additional uncertified pose-head regions'
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
