#!/usr/bin/env python3
"""Exact integer known-head feasibility versus a coarse candidate bank."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def interval(width, height, ratio, heads, k):
    p, q = ratio
    cw, ch = k * p, k * q
    lower_x = max([0] + [h[2] - cw for h in heads])
    upper_x = min([width - cw] + [h[0] for h in heads])
    lower_y = max([0] + [h[3] - ch for h in heads])
    upper_y = min([height - ch] + [h[1] for h in heads])
    if k < 1 or lower_x > upper_x or lower_y > upper_y:
        return None
    return [lower_x, lower_y, lower_x + cw, lower_y + ch]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows, counts, by_ratio = [], Counter(), defaultdict(Counter)
    for file in sorted(args.bank.glob('*.json')):
        source = json.loads(file.read_text())
        if 'tasks' not in source:
            continue
        width, height = source['width'], source['height']
        heads = [[math.floor(b[0] * width), math.floor(b[1] * height),
                  math.ceil(b[2] * width), math.ceil(b[3] * height)] for b in source['heads']]
        for task in source['tasks']:
            p, q = map(int, task['ratio'].split(':'))
            divisor = math.gcd(p, q)
            p, q = p // divisor, q // divisor
            maximum_k = min(width // p, height // q)
            largest = interval(width, height, (p, q), heads, maximum_k)
            aggressive = [interval(width, height, (p, q), heads, math.floor(maximum_k * s))
                          for s in [.9, .8, .7, .65]]
            aggressive = next((b for b in aggressive if b), None)
            for selection in task['selected']:
                old_contain = selection['detected_head_guarded_candidate'] is None
                mode = selection['mode']
                strict = largest if mode == 'minimal' else aggressive
                fallback = mode == 'maximal' and strict is None and largest is not None
                # Largest exact-ratio crop contains every smaller legal crop.
                # No possible legal size can exist if this origin interval is empty.
                status = ('infeasible_integer_geometry' if largest is None else
                          'requires_least_crop_fallback' if fallback else 'feasible_requested_amount')
                row = dict(id=source['id'], ratio=task['ratio'], mode=mode,
                           known_heads=len(heads), old_contain=old_contain, status=status,
                           witness_pixels=strict or largest, protected_head_pixels=heads,
                           source_size=[width, height], face_coverage_certified=False)
                if row['witness_pixels']:
                    x0, y0, x1, y1 = row['witness_pixels']
                    if (x1 - x0) * q != (y1 - y0) * p or not (
                            0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or any(
                            x0 > h[0] or y0 > h[1] or x1 < h[2] or y1 < h[3] for h in heads):
                        raise ValueError('integer feasibility witness failed independent guards')
                rows.append(row)
                keys = ['requests', status]
                if old_contain:
                    keys += ['old_contain', 'old_contain_' + status]
                if not heads:
                    keys += ['no_known_heads']
                for key in keys:
                    counts[key] += 1
                    by_ratio[task['ratio']][key] += 1
    result = dict(summary=dict(counts), by_ratio={r: dict(c) for r, c in by_ratio.items()},
        requests=rows, scope='integer geometry diagnostic for outward-rounded known heads; no face recall or artistic quality claim',
        units='exact integer source-pixel crop at reduced rational ratio; maximal scales .9/.8/.7/.65 with least-crop fallback',
        canonical_holdouts_opened=False, face_coverage_certified=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ['summary', 'by_ratio']}, indent=2))


if __name__ == '__main__':
    main()
