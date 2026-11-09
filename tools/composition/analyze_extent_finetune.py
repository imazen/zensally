#!/usr/bin/env python3
"""Summarize matched frozen and encoder-trained extent fits by family."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def paired(left,right,expected_families=8):
    def families(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['family_id']].append(row['iou'])
        return {key:statistics.mean(value) for key,value in grouped.items()}
    left,right = families(left),families(right)
    if left.keys() != right.keys() or len(left) != expected_families:
        raise ValueError('same registered development families required')
    values = [left[key]-right[key] for key in sorted(left)]
    rng = random.Random(31)
    draws = sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return dict(mean_iou_delta=statistics.mean(values),family_bootstrap_95_percentile=[draws[124],draws[4874]],
                families=len(values),selection_data=True,confirmatory=False)


def summarize(root,expected_arms):
    runs = json.loads((root/'summary.json').read_text())
    if {(r['arm'],r['seed']) for r in runs} != {(a,s) for a in expected_arms for s in [31,137,911]}:
        raise ValueError('registered complete arm/seed matrix required')
    result,rows = {},{}
    for arm in expected_arms:
        values = [r for r in runs if r['arm']==arm]
        result[arm] = dict(mean_metrics={key:statistics.mean(r['best_dev'][key] for r in values)
            for key in values[0]['best_dev']},selected_epochs=[r['best_epoch'] for r in values])
        rows[arm] = [row for r in values for row in json.loads(
            (root/(arm+'-seed'+str(r['seed']))/'best-predictions.json').read_text())]
    return result,rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    frozen,left = summarize(args.root/'extent-giou-fits',['aspect','image','permuted'])
    trained,right = summarize(args.root/'extent-joint-fits',['image','permuted'])
    audit = json.loads((args.root/'generalization.json').read_text())
    generalization = []
    for kind,arms in [('frozen',['aspect','image','permuted']),('trained',['image','permuted'])]:
        for arm in arms:
            for phase in ['selected','final']:
                values = [r for r in audit['runs'] if (r['kind'],r['arm'],r['phase'])==(kind,arm,phase)]
                if len(values) != 3:
                    raise ValueError('three audited checkpoints required per arm/phase')
                generalization.append(dict(kind=kind,arm=arm,phase=phase,roles={role:{key:statistics.mean(
                    r['roles'][role]['metrics'][key] for r in values) for key in values[0]['roles'][role]['metrics']}
                    for role in ['fit','dev']}))
    numerical = [json.loads(p.read_text()) for p in (args.root/'extent-joint-fits').glob('*/initial-numerics.json')]
    if len(numerical) != 6 or max(v['map_pixel_difference'] for v in numerical)>3:
        raise ValueError('six valid initial numerical receipts required')
    result = dict(frozen=frozen,trained=trained,generalization=generalization,
        paired_trained_image_vs=dict(frozen_image=paired(right['image'],left['image']),
            geometry=paired(right['image'],left['aspect']),permuted_trained=paired(right['image'],right['permuted'])),
        initial_max_map_edge_difference=max(v['map_pixel_difference'] for v in numerical),
        selected_replay_max_map_edge_difference=max(r.get('selected_replay_max_map_edge_difference',0) for r in audit['runs']),
        completed_fits=15,scope='eight reused synthetic development families; weak extents, not artistic crop quality',
        encoder_promoted=False,face_coverage_certified=False,new_test_inference=False,paid_model_api_calls=0)
    with (args.root/'analysis.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='generalization'},indent=2))


if __name__ == '__main__':
    main()
