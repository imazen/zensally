#!/usr/bin/env python3
"""Compare completed augmentation runs against their frozen unaugmented baseline."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_extent_finetune import paired,summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('baseline',type=Path)
    args = p.parse_args()
    directory = args.root/'extent-joint-augmented-fits'
    augmented,rows = summarize(directory,['image','permuted'])
    baseline,old = summarize(args.baseline/'extent-joint-fits',['image','permuted'])
    _,frozen = summarize(args.baseline/'extent-giou-fits',['aspect','image','permuted'])
    for arm in ['image','permuted']:
        for seed in [31,137,911]:
            name = arm+'-seed'+str(seed)
            before = json.loads((args.baseline/'extent-joint-fits'/name/'predictions-epoch-000.json').read_text())
            after = json.loads((directory/name/'predictions-epoch-000.json').read_text())
            if before != after:
                raise ValueError('matched epoch-zero predictions differ')
    audit = json.loads((args.root/'generalization.json').read_text())
    summary = []
    for arm in ['image','permuted']:
        for phase in ['selected','final']:
            values = [r for r in audit['runs'] if r['arm']==arm and r['phase']==phase]
            if len(values) != 3:
                raise ValueError('three audited seeds required')
            summary.append(dict(arm=arm,phase=phase,roles={role:{key:statistics.mean(
                r['roles'][role]['metrics'][key] for r in values) for key in values[0]['roles'][role]['metrics']}
                for role in ['fit','dev']}))
    numerical = [json.loads(p.read_text()) for p in directory.glob('*/initial-numerics.json')]
    if len(numerical) != 6 or max(r['map_pixel_difference'] for r in numerical)>3:
        raise ValueError('six initial numerical gates required')
    result = dict(augmented=augmented,baseline=baseline,epoch_zero_predictions_exact=True,
        generalization=summary,paired_augmented_image_vs=dict(unaugmented_image=paired(rows['image'],old['image']),
            augmented_permuted=paired(rows['image'],rows['permuted']),geometry=paired(rows['image'],frozen['aspect'])),
        initial_max_map_edge_difference=max(r['map_pixel_difference'] for r in numerical),
        selected_replay_max_map_edge_difference=max(r.get('selected_replay_max_map_edge_difference',0) for r in audit['runs']),
        completed_fits=6,scope='eight reused weak synthetic development families',
        encoder_promoted=False,face_coverage_certified=False,paid_model_api_calls=0,new_test_inference=False)
    with (args.root/'analysis.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
