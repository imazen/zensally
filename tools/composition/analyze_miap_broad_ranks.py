#!/usr/bin/env python3
"""Compare broader rank controls by frozen development corpus and family."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_unsplash_ranks import family_metrics,paired
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    protocol=json.loads((a.fits/'protocol.json').read_text());runs=json.loads((a.fits/'summary.json').read_text())
    progress=json.loads((a.fits/'progress.json').read_text())
    if progress['stage']!='complete' or len(runs)!=12 or {(r['arm'],r['seed']) for r in runs}!={(arm,seed) for arm in protocol['arms'] for seed in protocol['seeds']}:
        raise ValueError('all twelve broader fits required')
    rows={arm:{corpus:[] for corpus in ['old','new']} for arm in protocol['arms']}
    baseline={corpus:[] for corpus in ['old','new']};bindings=[]
    for run in runs:
        directory=a.fits/(run['arm']+'-seed'+str(run['seed']));path=directory/'best.pt'
        if sha(path)!=run['best_weights_sha256'] or not run['selected_replay_exact'] or not run['source_query_visits_exact']:
            raise ValueError('selected broader state/replay differs')
        prediction=json.loads((directory/'best-predictions.json').read_text())
        if len(prediction)!=2552 or len({r['id'] for r in prediction})!=116:raise ValueError('all116 development sources and22 requests required')
        for corpus in ['old','new']:
            rows[run['arm']][corpus]+=[r for r in prediction if r['corpus']==corpus]
        if run['arm']=='trainable-image':
            initial=json.loads((directory/'predictions-000.json').read_text())
            for corpus in ['old','new']:baseline[corpus]+=[r for r in initial if r['corpus']==corpus]
        bindings.append(dict(arm=run['arm'],seed=run['seed'],best_epoch=run['best_epoch'],path=str(path.resolve()),
            sha256=sha(path),selected_development=run['best_dev']))
    families={arm:{corpus:{scope:family_metrics(values,scope) for scope in ['all','essential','secondary']} for corpus,values in corpora.items()} for arm,corpora in rows.items()}
    initial_families={corpus:{scope:family_metrics(values,scope) for scope in ['all','essential','secondary']} for corpus,values in baseline.items()}
    if any(len(families[arm][corpus]['essential'])!=count for arm in families for corpus,count in [('old',21),('new',91)]):
        raise ValueError('registered development-family supports differ')
    comparisons={control:{corpus:{scope:paired(families['trainable-image'][corpus][scope],
        initial_families[corpus][scope] if control=='initializer' else families[control][corpus][scope])
        for scope in ['all','essential','secondary']} for corpus in ['old','new']} for control in ['geometry-only','permuted-image','frozen-image','initializer']}
    qualifies=all(comparisons[c]['new']['essential']['teacher_regret']['family_bootstrap95'][1]<0 for c in ['geometry-only','permuted-image','initializer'])
    candidate=min((r for r in runs if r['arm'] in ['trainable-image','frozen-image']),key=lambda r:(
        sum(r['best_dev'][c]['essential']['teacher_regret'] for c in ['old','new'])/2,
        -sum(r['best_dev'][c]['essential']['strong_pair_accuracy'] for c in ['old','new'])/2,r['best_epoch'],r['arm'],r['seed']))
    report=dict(fit_protocol_sha256=sha(a.fits/'protocol.json'),fit_summary_sha256=sha(a.fits/'summary.json'),
        arms={arm:dict(selected_epochs=[r['best_epoch'] for r in runs if r['arm']==arm],
            mean_selected={corpus:{scope:{key:statistics.mean(r['best_dev'][corpus][scope][key] for r in runs if r['arm']==arm)
                for key in ['teacher_regret','strong_pair_accuracy','exact_top1','within_3_map_pixels']}
                for scope in ['all','essential','secondary']} for corpus in ['old','new']}) for arm in protocol['arms']},
        initializer={corpus:{scope:{key:statistics.mean(v[key] for v in values.values()) for key in next(iter(values.values()))} for scope,values in scopes.items()} for corpus,scopes in initial_families.items()},
        paired_trainable=comparisons,family_metrics=families,models=bindings,permutation_coverage=protocol['permutation_coverage'],
        registered_new_photo_agreement_gate_passed=qualifies,candidate_for_review=dict(arm=candidate['arm'],seed=candidate['seed'],weights_sha256=candidate['best_weights_sha256']),
        all12_selected_native_replays_exact=True,fit_sources=369,old_development_sources=25,new_development_sources=91,
        new_test_inference=False,human_artistic_quality_measured=False,face_coverage_certified=False,model_promoted=False,paid_model_api_calls=0)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['arms','initializer','registered_new_photo_agreement_gate_passed','candidate_for_review','permutation_coverage']},indent=2))


if __name__=='__main__':main()
