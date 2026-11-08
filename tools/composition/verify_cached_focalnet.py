#!/usr/bin/env python3
"""Whole-bank rectangle gate for cached analysis, retaining only scalar errors."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from cached_focalnet import CachedContext, prepared, session
from fit_heads import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path)
    parser.add_argument('split', type=Path)
    parser.add_argument('auxiliary', type=Path)
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--analysis-threads',type=int,choices=[1,2,4],default=1)
    parser.add_argument('--query-threads',type=int,choices=[1,2,4])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path.insert(0, str(args.models/'focalnet-source/src'))
    from focalnet.imaging import Letterbox, restore_map
    from focalnet.crop import Crop, score_crop
    torch.set_num_threads(2)
    cache = args.auxiliary/'auxiliary.pt'
    auxiliary = torch.load(cache, map_location='cpu', weights_only=False)
    model_path = args.models/'focalnet/focalnet-human.onnx'
    full = session(model_path, optimized=args.optimized)
    context = CachedContext(args.split, optimized=args.optimized,
                            threads=args.analysis_threads, query_threads=args.query_threads)
    result = dict(sources=0, requests=0, choice_changes=0, action_changes=0,
        max_map_pixel_edge_movement=0., edge_movements_over_three=0,
        max_abs_score_error=0., max_abs_importance_error=0., stale_token_rejections=0,
        exact_score_arrays=0, compared_arrays=0)
    rows = []
    previous_token = None
    for i, source in enumerate(auxiliary['sources']):
        bank_path = args.bank/(source['id']+'.json')
        bank = json.loads(bank_path.read_text())
        image, content = prepared(auxiliary, i)
        token = context.analyze(image, content, (source['id'], source['rgb_sha256']))
        empty = np.zeros((1, 128, 4), np.float32)
        if previous_token is not None:
            try:
                context.score(empty, previous_token)
            except ValueError:
                result['stale_token_rejections'] += 1
            else:
                raise ValueError('stale source cache accepted')
        previous_token = token
        width, height = bank['width'], bank['height']
        valid = content[0]
        x0, y0, x1, y1 = np.rint(valid*256).astype(int)
        letterbox = Letterbox(256, x0, y0, x1-x0, y1-y0)
        for task in bank['tasks']:
            boxes = np.zeros((1, 128, 4), np.float32)
            for j, candidate in enumerate(task['candidates']):
                boxes[0, j] = candidate['box']
            expected_map, expected_scores = full.run(['importance','crop_scores'],
                dict(image=image, boxes=boxes, content=content))
            actual_scores = context.score(boxes, token)
            actual_map = context.outputs['importance']
            result['compared_arrays'] += 1
            result['exact_score_arrays'] += int(np.array_equal(actual_scores, expected_scores))
            result['max_abs_score_error'] = max(result['max_abs_score_error'],float(np.max(np.abs(actual_scores-expected_scores))))
            result['max_abs_importance_error'] = max(result['max_abs_importance_error'],float(np.max(np.abs(actual_map-expected_map))))
            retentions = []
            for importance in [expected_map, actual_map]:
                restored = restore_map(importance[0,0], letterbox)
                retentions.append([score_crop(restored, Crop(b[0],b[1],b[2]-b[0],b[3]-b[1],0),width,height)
                                   for b in [c['pixel_box'] for c in task['candidates']]])
            p, q = map(int, task['ratio'].split(':'))
            maximum = min(width//p, height//q)
            for mode in ['minimal','maximal']:
                eligible = [j for j,c in enumerate(task['candidates']) if
                    (c['pixel_box'][2]-c['pixel_box'][0] == maximum*p if mode=='minimal'
                     else c['pixel_box'][2]-c['pixel_box'][0] < maximum*p)]
                if not eligible and mode=='maximal':
                    eligible = list(range(len(task['candidates'])))
                chosen = []
                for retention, scores in zip(retentions,[expected_scores,actual_scores]):
                    if eligible:
                        best = max(retention[j] for j in eligible)
                        chosen.append(max((j for j in eligible if retention[j]>=best-.05),key=lambda j:float(scores[0,j])))
                    else:
                        chosen.append(None)
                result['requests'] += 1
                changed = chosen[0]!=chosen[1]
                result['choice_changes'] += int(changed)
                action_changed = (chosen[0] is None)!=(chosen[1] is None)
                result['action_changes'] += int(action_changed)
                movement = 0.
                if changed and not action_changed:
                    left,right=[task['candidates'][j]['box'] for j in chosen]
                    movement = 128*max(abs(a-b) for a,b in zip(left,right))
                result['max_map_pixel_edge_movement']=max(result['max_map_pixel_edge_movement'],movement)
                result['edge_movements_over_three']+=int(movement>3)
                rows.append(dict(id=source['id'],ratio=task['ratio'],mode=mode,full_choice=chosen[0],cached_choice=chosen[1],map_pixel_movement=movement))
        result['sources'] += 1
    result.update(optimized=args.optimized,analysis_threads=args.analysis_threads,
        query_threads=args.query_threads or args.analysis_threads,source_cache_sha256=sha(cache),
        model_sha256=sha(model_path),split_identity_sha256=sha(args.split/'identity.json'),
        context_retained_tensor_payload_bytes=context.retained_tensor_bytes(),
        diagnostic_maps_saved=False,face_coverage_certified=False,new_model_quality_claim=False)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    with (args.output/'choices.json').open('x') as stream:
        json.dump(rows,stream,indent=2)
    print(json.dumps(result,indent=2))
    if result['edge_movements_over_three'] or result['action_changes']:
        raise ValueError('cached graph failed unchanged three-map-pixel edge/action gate')


if __name__ == '__main__':
    main()
