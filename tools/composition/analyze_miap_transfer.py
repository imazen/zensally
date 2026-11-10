#!/usr/bin/env python3
"""Compare frozen transfer selections; auxiliary gains cannot qualify crop quality."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_unsplash_ranks import family_metrics, paired
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    runs=json.loads((a.fits/'summary.json').read_text());protocol=json.loads((a.fits/'protocol.json').read_text());progress=json.loads((a.fits/'progress.json').read_text())
    expected={(arm,seed) for arm in protocol['arms'] for seed in protocol['seeds']}
    if progress['stage']!='complete' or len(runs)!=12 or {(r['arm'],r['seed']) for r in runs}!=expected:
        raise ValueError('all twelve transfer fits must be terminal')
    rows={arm:[] for arm in protocol['arms']};bindings=[];final_aux={arm:[] for arm in protocol['arms']}
    for run in runs:
        root=a.fits/(run['arm']+'-seed'+str(run['seed']));path=root/'best.pt'
        if sha(path)!=run['best_weights_sha256'] or not run['selected_replay_exact'] or not run['source_query_visits_exact']:
            raise ValueError('selected model/replay binding differs')
        prediction=json.loads((root/'best-predictions.json').read_text())
        if len(prediction)!=550 or len({r['id'] for r in prediction})!=25:raise ValueError('all25 development sources and22 queries required')
        rows[run['arm']]+=prediction
        history=json.loads((root/'history.json').read_text())
        if len(history)!=45 or history[-1]['epoch']!=44:raise ValueError('all registered epochs required')
        final_aux[run['arm']].append(history[-1]['person_development'])
        bindings.append(dict(arm=run['arm'],seed=run['seed'],best_epoch=run['best_epoch'],weights_sha256=sha(path),
            selected_development=run['best_dev'],selected_person_development=run['person_development'],final_person_development=history[-1]['person_development']))
    families={arm:{scope:family_metrics(values,scope) for scope in ['all','essential','secondary']} for arm,values in rows.items()}
    if any(len(families[arm]['essential'])!=21 for arm in families):raise ValueError('fixed21-family support differs')
    comparisons={control:{scope:paired(families['joint-image'][scope],families[control][scope]) for scope in ['all','essential','secondary']}
        for control in ['rank-only','joint-permuted','joint-geometry']}
    qualifies=all(comparisons[control]['essential']['teacher_regret']['family_bootstrap95'][1]<0 for control in ['rank-only','joint-permuted'])
    report=dict(fit_protocol_sha256=sha(a.fits/'protocol.json'),fit_summary_sha256=sha(a.fits/'summary.json'),
        arms={arm:dict(selected_epochs=[r['best_epoch'] for r in runs if r['arm']==arm],
            mean_selected_essential={key:statistics.mean(r['best_dev']['essential'][key] for r in runs if r['arm']==arm) for key in ['teacher_regret','strong_pair_accuracy','exact_top1','within_3_map_pixels']},
            mean_final_person_development={key:statistics.mean(row[key] for row in final_aux[arm]) for key in final_aux[arm][0]}) for arm in protocol['arms']},
        paired_image=comparisons,family_metrics=families,models=bindings,
        registered_crop_transfer_gate_passed=qualifies,all12_selected_native_replays_exact=True,
        development_sources=25,development_families=21,auxiliary_fit_sources=275,auxiliary_development_sources=91,
        unmoved_auxiliary=protocol['unmoved_auxiliary'],new_test_inference=False,human_artistic_quality_measured=False,
        face_coverage_certified=False,model_promoted=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['arms','registered_crop_transfer_gate_passed','all12_selected_native_replays_exact','unmoved_auxiliary']},indent=2))


if __name__=='__main__':main()
