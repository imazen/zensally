#!/usr/bin/env python3
"""Report all fixed-model supplementary comparisons without selecting a new model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from analyze_extent_finetune import paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    evaluation = json.loads((args.root/'evaluation.json').read_text())
    qualification = json.loads((args.root/'qualification.json').read_text())
    if evaluation['qualified_models']!=48 or evaluation['evaluated_sources']!=15 or evaluation['unknown_sources']:
        raise ValueError('all48 fixed models and fifteen current references required')
    grouped = defaultdict(list)
    for row in evaluation['models']:
        grouped[(row['study'],row['arm'])].append(row)
    result,all_rows = {},{}
    for (study,arm),runs in grouped.items():
        if {r['seed'] for r in runs}!={31,137,911}:
            raise ValueError('all three preselected seeds required')
        rows = [p for r in runs for p in r['predictions']]
        all_rows[(study,arm)] = rows
        strata = {}
        for name,predicate,expected in [('new_people',lambda p:int(p['id'].split(':')[1])>=12000,8),
            ('prior_groups',lambda p:int(p['id'].split(':')[1])<12000 and p['id']!='imazen26:11440',6),
            ('wildlife',lambda p:p['id']=='imazen26:11440',1)]:
            chosen = [r for r in rows if predicate(r)]
            if len({r['family_id'] for r in chosen})!=expected:
                raise ValueError('registered supplementary source stratum differs')
            strata[name] = dict(sources=expected,mean_iou=statistics.mean(r['iou'] for r in chosen))
        result.setdefault(study,{})[arm] = dict(mean_metrics={key:statistics.mean(r['metrics'][key] for r in runs)
            for key in runs[0]['metrics']},strata=strata)
    comparisons = {}
    for study in result:
        left = all_rows[(study,'image')]
        comparisons[study] = dict(vs_permuted=paired(left,all_rows[(study,'permuted')],expected_families=15))
        geometry_study = 'small-giou' if study in ['small-trained','small-augmented'] else study
        comparisons[study]['vs_geometry'] = paired(left,all_rows[(geometry_study,'aspect')],expected_families=15)
    comparisons['small-augmented']['vs_unaugmented'] = paired(all_rows[('small-augmented','image')],
        all_rows[('small-trained','image')],expected_families=15)
    if any(value['families']!=15 for group in comparisons.values() for value in group.values()):
        raise ValueError('paired uncertainty must report fifteen source families')
    output = dict(studies=result,paired_family_comparisons=comparisons,
        mean_box_control=evaluation['fit_mean_control'],reference_sources=15,
        valid_reference_boxes=sum(r['valid_boxes'] for r in evaluation['reference_receipts']),
        qualified_models=48,max_qualification_map_edge_difference=qualification['max_map_edge_difference'],
        matched_mse_geometry_exact=qualification['matched_mse_geometry_exact'],
        scope='fourteen people scenes and one wildlife scene; supplementary development, weak query unions, not artistic quality',
        model_selection_changed=False,new_gradients=False,model_promoted=False,
        face_coverage_certified=False,new_test_inference=False,paid_model_api_calls=0)
    with (args.root/'analysis.json').open('x') as stream:
        stream.write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(means={study:{arm:value['mean_metrics']['mean_iou'] for arm,value in arms.items()}
        for study,arms in result.items()},paired=comparisons,qualification=output['max_qualification_map_edge_difference']),indent=2))


if __name__ == '__main__':
    main()
