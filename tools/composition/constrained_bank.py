#!/usr/bin/env python3
"""Score exact-ratio integer crops within known-head origin intervals."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from audit_crop_feasibility import interval
from fit_heads import sha


def candidates(width, height, ratio, heads):
    p, q = ratio
    divisor = math.gcd(p, q)
    p, q = p // divisor, q // divisor
    maximum = min(width // p, height // q)
    result = []
    for k in sorted({math.floor(maximum * s) for s in [1., .9, .8, .7, .65]}, reverse=True):
        first = interval(width, height, (p, q), heads, k)
        if first is None:
            continue
        cw, ch = k * p, k * q
        lx, ly = first[:2]
        hx = min([width - cw] + [h[0] for h in heads])
        hy = min([height - ch] + [h[1] for h in heads])
        for iy in range(5):
            for ix in range(5):
                x, y = lx + (hx - lx) * ix // 4, ly + (hy - ly) * iy // 4
                box = (x, y, x + cw, y + ch)
                if box not in result:
                    result.append(box)
    if len(result) > 128:
        raise ValueError('constrained candidate bank exceeded fixed slots')
    return result, maximum * p, maximum * q


def main():
    import numpy as np
    import onnxruntime as ort
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path)
    parser.add_argument('auxiliary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path.insert(0, str(args.models / 'focalnet-source/src'))
    from focalnet.imaging import Letterbox, restore_map
    from focalnet.crop import Crop, score_crop
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model_path = args.models / 'focalnet/focalnet-human.onnx'
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    aux_path = args.auxiliary / 'auxiliary.pt'
    auxiliary = torch.load(aux_path, map_location='cpu', weights_only=False)
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    counts = Counter()
    for i, source in enumerate(auxiliary['sources']):
        original_path = args.models / 'focalnet-bank' / (source['id'] + '.json')
        old = json.loads(original_path.read_text())
        width, height = old['width'], old['height']
        heads = [[math.floor(b[0] * width), math.floor(b[1] * height),
                  math.ceil(b[2] * width), math.ceil(b[3] * height)] for b in old['heads']]
        valid = auxiliary['packed']['valid_rect'][i].numpy()
        x0, y0, x1, y1 = np.rint(valid * 256).astype(int)
        image = np.zeros((1, 3, 256, 256), np.float32)
        image[0, :, y0:y1, x0:x1] = ((auxiliary['image'][i].numpy() - mean) / std)[:, y0:y1, x0:x1]
        tasks = []
        for task in old['tasks']:
            ratio = tuple(map(int, task['ratio'].split(':')))
            boxes, maximum_w, maximum_h = candidates(width, height, ratio, heads)
            values = []
            if boxes:
                normalized = np.array(boxes, np.float64) / [width, height, width, height]
                padded = np.zeros((1, 128, 4), np.float32)
                padded[0, :len(boxes)] = normalized
                importance, scores = session.run(['importance', 'crop_scores'],
                    dict(image=image, boxes=padded, content=valid[None]))
                if not np.isfinite(importance).all() or not np.isfinite(scores).all():
                    raise ValueError('nonfinite constrained crop model output')
                # Required transient network output; no diagnostic map is saved.
                restored = restore_map(importance[0, 0], Letterbox(256, x0, y0, x1 - x0, y1 - y0))
                for j, (l, t, r, b) in enumerate(boxes):
                    if ((r - l) * ratio[1] != (b - t) * ratio[0] or
                            not (0 <= l < r <= width and 0 <= t < b <= height) or any(
                            l > h[0] or t > h[1] or r < h[2] or b < h[3] for h in heads)):
                        raise ValueError('integer candidate failed independent boundary guards')
                    values.append(dict(box=normalized[j].tolist(), pixel_box=[l, t, r, b],
                        score=float(scores[0, j]), contains_detected_heads=True,
                        retained_importance=float(score_crop(restored, Crop(l, t, r - l, b - t, 0), width, height))))
            selected = []
            for previous in task['selected']:
                mode = previous['mode']
                eligible = [j for j, b in enumerate(boxes) if
                    ((b[2] - b[0] == maximum_w and b[3] - b[1] == maximum_h) if mode == 'minimal'
                     else (b[2] - b[0] < maximum_w))]
                fallback = bool(boxes) and not eligible and mode == 'maximal'
                if fallback:
                    eligible = list(range(len(boxes)))
                chosen = None
                if eligible:
                    retention = max(values[j]['retained_importance'] for j in eligible)
                    chosen = max((j for j in eligible if values[j]['retained_importance'] >= retention - .05),
                                 key=lambda j: values[j]['score'])
                counts['requests'] += 1
                counts['contain' if chosen is None else 'crop'] += 1
                counts['least_crop_fallback'] += int(fallback)
                if previous['detected_head_guarded_candidate'] is None and chosen is not None:
                    counts['recovered_old_contain'] += 1
                selected.append(dict(mode=mode, raw_candidate=chosen,
                    detected_head_guarded_candidate=chosen, minimal_fallback=fallback,
                    guarded_action='contain' if chosen is None else 'research_crop',
                    face_coverage_certified=False, deployable=False))
            tasks.append(dict(ratio=task['ratio'], candidates=values, selected=selected))
        row = {k: old[k] for k in ['id', 'family_id', 'use', 'width', 'height', 'heads']}
        row.update(tasks=tasks, protected_head_pixels=heads, original_bank_sha256=sha(original_path),
                   diagnostic_maps_saved=False, face_coverage_certified=False, deployable=False)
        with (args.output / (source['id'] + '.json')).open('x') as stream:
            json.dump(row, stream)
            stream.write('\n')
        counts['sources'] += 1
        print(json.dumps(dict(completed=counts['sources'], id=source['id'])), flush=True)
    summary = dict(counts, model_sha256=sha(model_path), source_cache_sha256=sha(aux_path),
        candidate_contract='exact integer ratio; outward-rounded known heads; five feasible origin anchors per axis; maximal falls back only when no aggressive scale fits',
        scoring='unchanged retained-importance tolerance .05 then FocalNet score',
        diagnostic_maps_saved=False, face_coverage_certified=False, artistic_quality_established=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
