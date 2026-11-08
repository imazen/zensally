#!/usr/bin/env python3
"""Analyze completed controls and freeze all model identities before TEST scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import subprocess

from artifact_identity import sha


def family_metrics(rows, scope):
    grouped = defaultdict(list)
    for row in rows:
        if row['query_valid'] and (scope == 'all' or (row['query'] < 18) == (scope == 'essential')):
            grouped[(row['family_id'], row['id'])].append(row)
    families = defaultdict(list)
    for (family, _), group in grouped.items():
        families[family].append(dict(teacher_regret=statistics.mean(r['teacher_regret'] for r in group),
            strong_pair_accuracy=statistics.mean(r['strong_pair_accuracy'] for r in group),
            exact_top1=statistics.mean(r['selected'] == r['teacher'] for r in group),
            within_3_map_pixels=statistics.mean(r['max_edge_map_pixels'] <= 3 for r in group)))
    return {family: {key:statistics.mean(r[key] for r in values) for key in values[0]}
            for family, values in families.items()}


def paired(left, right):
    if left.keys() != right.keys():
        raise ValueError('paired comparison families differ')
    result = {}
    for key in next(iter(left.values())):
        delta = [left[f][key]-right[f][key] for f in sorted(left)]
        rng = random.Random(31)
        samples = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(5000))
        result[key] = dict(delta=statistics.mean(delta), family_bootstrap95=[samples[124], samples[4874]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['fits', 'registry', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.fits/'protocol.json').read_text())
    runs = json.loads((args.fits/'summary.json').read_text())
    progress = json.loads((args.fits/'progress.json').read_text())
    if (progress['stage'] != 'complete' or progress['completed_runs'] != 12
            or {(r['arm'], r['seed']) for r in runs} !=
               {(a, s) for a in protocol['arms'] for s in [31, 137, 911]}):
        raise ValueError('all twelve registered fits required')
    rows, bindings = defaultdict(list), []
    for run in runs:
        directory = args.fits/(run['arm']+'-seed'+str(run['seed']))
        path = directory/'best.pt'
        if sha(path) != run['best_weights_sha256']:
            raise ValueError('selected model state differs')
        values = json.loads((directory/'best-predictions.json').read_text())
        if len(values) != 550 or len({v['id'] for v in values}) != 25:
            raise ValueError('complete development source/query predictions required')
        rows[run['arm']].extend(values)
        bindings.append(dict(arm=run['arm'], seed=run['seed'], path=str(path.resolve()),
            sha256=sha(path), bytes=path.stat().st_size, best_epoch=run['best_epoch'],
            selected_development_metrics=run['best_dev']))
    families = {arm:{scope:family_metrics(values, scope) for scope in ['all', 'essential', 'secondary']}
                for arm,values in rows.items()}
    if any(len(families[arm]['all']) != 21 for arm in families):
        raise ValueError('fixed21 development-family support differs')
    comparisons = {control:{scope:paired(families['trainable-image'][scope], families[control][scope])
        for scope in ['all', 'essential', 'secondary']} for control in ['geometry-only', 'permuted-image', 'frozen-image']}
    report = dict(arms={arm:dict(mean_selected_metrics={key:statistics.mean(r['best_dev']['all'][key]
        for r in runs if r['arm'] == arm) for key in runs[0]['best_dev']['all']
        if key not in ['sources', 'families', 'valid_queries']}, family_metrics=families[arm]) for arm in families},
        paired_trainable=comparisons, moved_across_families=protocol['moved_across_families'],
        development_sources=25, development_families=21, detector_person_positive_development_sources=3,
        new_test_composition_scores_opened=0, human_artistic_quality_measured=False, face_coverage_certified=False,
        intervals_exploratory=True, quality_model_promoted=False)
    candidates = [r for r in runs if r['arm'] in ['frozen-image', 'trainable-image']]
    candidate = min(candidates, key=lambda r:(r['best_dev']['all']['teacher_regret'],
        -r['best_dev']['all']['strong_pair_accuracy'], r['best_epoch'], r['arm'], r['seed']))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'development-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    plan = dict(schema='unsplash-final-rank-comparison-v1',
        build_commit=subprocess.check_output(['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], text=True).strip(),
        registry_sha256=sha(args.registry), fit_protocol_sha256=sha(args.fits/'protocol.json'),
        fit_summary_sha256=sha(args.fits/'summary.json'), models=bindings,
        candidate_for_export=dict(arm=candidate['arm'], seed=candidate['seed'],
            weights_sha256=candidate['best_weights_sha256']),
        selection_data='fixed development only; no TEST crop reference/score read',
        test_sources=25, test_families=22, test_detector_positive_people_sources=8,
        evaluation='all12 frozen models, same metrics/ratio scopes/family weighting; no retuning or model reselection',
        student_source_sha256=sha(Path('training/local_rank_distillation.py')),
        region_source_sha256=sha(Path('training/crop_regions.py')),
        test_quality_protocol='weak local FocalNet relative ranking; not human acceptability or complete face coverage',
        paid_model_api_calls=0)
    (args.output/'test-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(arms={arm:r['mean_selected_metrics'] for arm,r in report['arms'].items()},
        paired_trainable_all={arm:value['all'] for arm,value in comparisons.items()},
        candidate_for_export=plan['candidate_for_export'], test_plan_sha256=sha(args.output/'test-plan.json'))))


if __name__ == '__main__':
    main()
