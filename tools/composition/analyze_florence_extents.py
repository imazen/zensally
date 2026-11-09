#!/usr/bin/env python3
"""Summarize frozen phrase-target fits without inference or new selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from artifact_identity import sha


def study(root):
    runs = json.loads((root/'summary.json').read_text())
    if len(runs) != 9 or {(r['arm'],r['seed']) for r in runs} != {
            (a,s) for a in ['aspect','image','permuted'] for s in [31,137,911]}:
        raise ValueError('all nine registered fits required')
    rows, metrics = defaultdict(list), defaultdict(list)
    for run in runs:
        directory = root/(run['arm']+'-seed'+str(run['seed']))
        rows[run['arm']].extend(json.loads((directory/'best-predictions.json').read_text()))
        metrics[run['arm']].append(run['best_dev'])
    result = dict(arms={a:{k:statistics.mean(m[k] for m in values) for k in values[0]}
                      for a,values in metrics.items()},
        fit_mean=json.loads((root/'fit-mean-control.json').read_text())['metrics'],
        selected_epochs={a:[r['best_epoch'] for r in runs if r['arm']==a] for a in metrics},
        protocol_sha256=sha(root/'protocol.json'))
    family = {}
    for arm,values in rows.items():
        grouped = defaultdict(list)
        for row in values:
            grouped[row['family_id']].append(row['iou'])
        family[arm] = {k:statistics.mean(v) for k,v in grouped.items()}
    paired = {}
    for control in ['aspect','permuted']:
        if family['image'].keys() != family[control].keys() or len(family[control]) != 8:
            raise ValueError('paired eight-family development set differs')
        differences = [family['image'][f]-family[control][f] for f in sorted(family['image'])]
        rng = random.Random(31)
        draws = sorted(statistics.mean(rng.choices(differences,k=8)) for _ in range(5000))
        paired[control] = dict(mean_iou_delta=statistics.mean(differences), families=8,
            family_bootstrap_95_percentile=[draws[124],draws[4874]],
            selection_data=True, confirmatory=False)
    result['paired_image_vs'] = paired
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    small, focal = study(args.root/'extent-fits'), study(args.root/'focalnet/extent-fits')
    for seed in [31,137,911]:
        name = 'aspect-seed'+str(seed)
        for file in ['history.json','best-predictions.json']:
            if json.loads((args.root/'extent-fits'/name/file).read_text()) != json.loads(
                    (args.root/'focalnet/extent-fits'/name/file).read_text()):
                raise ValueError('matched geometry-only control differs across descriptor studies')
    admission = json.loads((args.root/'extent-cache/admission.json').read_text())
    result = dict(small=small, focal=focal, matched_aspect_controls_exact=True,
        admitted_fit_sources=admission['fit_eligible'], development_sources=admission['dev_eligible'],
        valid_teacher_boxes=admission['valid_boxes'], source_plan_sha256=sha(args.root/'source-plan.json'),
        scope='eight reused synthetic development families; noisy query-conditioned extent targets, not artistic crop quality',
        model_promoted=False, human_rated=False, face_coverage_certified=False,
        canonical_holdouts_opened=False, paid_model_api_calls=0)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
