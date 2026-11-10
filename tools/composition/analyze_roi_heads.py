#!/usr/bin/env python3
"""Paired source/family summaries of all ROI head seeds, without inference."""
import argparse
from collections import defaultdict
import json
import random
import statistics
from pathlib import Path


def family_values(rows):
    sources = defaultdict(list)
    for row in rows:
        sources[(row['family_id'], row['source_id'])].append(row)
    families = defaultdict(list)
    keys = ['iou', 'edge_mae', 'edge_mse', 'objective', 'contains_weak_target']
    for (family, _), values in sources.items():
        families[family].append({k: statistics.mean(float(v[k]) for v in values) for k in keys})
    return {family: {k: statistics.mean(v[k] for v in values) for k in keys}
            for family, values in families.items()}


def paired(left, right):
    if left.keys() != right.keys():
        raise ValueError('comparison families differ')
    result = {}
    for metric in ['iou', 'edge_mae', 'contains_weak_target']:
        differences = [left[f][metric]-right[f][metric] for f in sorted(left)]
        rng = random.Random(31)
        samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
        result[metric] = dict(delta=statistics.mean(differences), family_bootstrap95=[samples[124], samples[4874]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fits', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.fits/'summary.json').read_text())
    if {(r['arm'], r['seed']) for r in runs} != {(a, s) for a in ['image', 'geometry', 'permuted'] for s in [31, 137, 911]}:
        raise ValueError('registered nine fits required')
    arms, metrics = defaultdict(list), defaultdict(list)
    for run in runs:
        rows = json.loads((args.fits/(run['arm']+'-seed'+str(run['seed']))/'best-predictions.json').read_text())
        arms[run['arm']].extend(rows)
        metrics[run['arm']].append(run['best_dev'])
    controls = json.loads((args.fits/'fit-mean-control.json').read_text())
    arms['fit_mean'] = controls['predictions']
    families = {arm: family_values(rows) for arm, rows in arms.items()}
    if any(len(values) != 7 for values in families.values()):
        raise ValueError('fixed seven-development-family support differs')
    report = dict(families=7, roi_count=12, arms={arm: dict(
        mean_selected_metrics={k: statistics.mean(m[k] for m in values) for k in values[0]},
        family_metrics=families[arm]) for arm, values in metrics.items()},
        fit_mean_metrics=controls['metrics'],
        paired_image={arm: paired(families['image'], families[arm]) for arm in ['geometry', 'permuted', 'fit_mean']},
        seeds=runs, selection_data=True, confirmatory=False, face_coverage_certified=False,
        teacher_targets_complete=False, artistic_quality_measured=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['families', 'fit_mean_metrics', 'paired_image']}))


if __name__ == '__main__':
    main()
