#!/usr/bin/env python3
"""Reuse finite weak same-story endpoints and measured features without new labels."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import write
from refinement_inputs import accepted_options


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['support','direct','attention','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    expected=[(a.support/'SHA256SUMS.json','ae18aeeaad6f692c4d832e67e2a9071cc6aaf46751cafec8bc3e19c515490dcf'),(a.direct,'2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472'),(a.attention,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('registered input identity differs')
    for member in ['predictions.jsonl','protocol.json']:verify_member(a.support,member)
    d=torch.load(a.direct,map_location='cpu',weights_only=False);c=torch.load(a.attention,map_location='cpu',weights_only=False);identity=lambda s:(s['id'],s['role'],s['family_id'],s['rgb_sha256'],s['native_size'])
    if list(map(identity,d['sources']))!=list(map(identity,c['sources'])) or d['ratios']!=c['ratios']:raise ValueError('feature/source identity differs')
    rows=[json.loads(line) for line in (a.support/'predictions.jsonl').read_text().splitlines()];lookup={(r['id'],r['ratio_index']):[v for v in r['exact_options'] if v['interval_safe'] and v['same_story']] for r in rows};maximum=max(map(len,lookup.values()));target=torch.zeros(72,11,maximum,2,4);valid=torch.zeros(72,11,maximum,dtype=torch.bool)
    for i,s in enumerate(c['sources']):
        scale=torch.tensor([*s['native_size'],*s['native_size']])
        for ri in range(11):
            for k,row in enumerate(lookup[s['id'],ri]):target[i,ri,k]=torch.tensor([row['outer'],row['inner']])/scale;valid[i,ri,k]=True
    fit=[i for i,s in enumerate(c['sources']) if s['role']=='fit'];dev=[i for i,s in enumerate(c['sources']) if s['role']=='dev']
    if len(fit)!=48 or len(dev)!=24 or {c['sources'][i]['family_id'] for i in fit}&{c['sources'][i]['family_id'] for i in dev}:raise ValueError('fixed family roles differ')
    eligible=c['eligible'][...,None];denominator=eligible.sum(-2).clamp_min(1)
    def summary(values):return ((values*eligible).sum(-2)/denominator).reshape(72,11,2,32).mean(-2)
    measured=summary(c['extras']);permuted=summary(c['permuted_extras']);mean=measured[fit].mean((0,1));std=measured[fit].std((0,1),correction=0).clamp_min(.001);visual=(d['visual']-d['mean'])/d['std'];result=dict(sources=c['sources'],ratios=c['ratios'],visual=visual,visual_mean=d['mean'],visual_std=d['std'],attention=(measured-mean)/std,permuted_attention=(permuted-mean)/std,attention_mean=mean,attention_std=std,targets=target,target_valid=valid,references=d['references'],reference_valid=d['reference_valid'],head_rois=d['head_rois'])
    reference_paths=json.loads((a.support/'protocol.json').read_text())['references'];roots={Path(path).parents[2] for path in reference_paths if Path(path).name=='zones.json'}
    if len(roots)!=1:raise ValueError('one pinned cached reference root required')
    native_scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in c['sources']])[:,None,None,:];native=(c['query_boxes']*native_scale).round().long();reference_options,head_boxes,reference_ids,replayed=accepted_options(dict(data=c,native=native),next(iter(roots)))
    if reference_ids!=reference_paths:raise ValueError('cached reference identities differ from support study')
    result.update(reference_options=reference_options,heads=head_boxes,reference_positive_replayed_queries=replayed)
    if not torch.equal(c['reference_valid'],d['reference_valid']):raise ValueError('independent reference mask alignment differs')
    if any(not torch.isfinite(result[k]).all() for k in ['visual','attention','permuted_attention','targets']):raise ValueError('nonfinite training tensor')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'protocol.json',dict(inputs={str(path):digest for path,digest in expected},labels='existing finite weak exact endpoint combinations only; no new excellent or interior labels',normalization='aligned FIT-only',api_calls=0,test_inference=False,new_teacher_inference=False,model_promoted=False));torch.save(result,a.output/'cache.pt');report=dict(sources=72,fit=48,dev=24,max_target_combinations=maximum,fit_pair_requests=int(valid[fit].any(-1).sum()),dev_pair_requests=int(valid[dev].any(-1).sum()),fit_pair_sources=int(valid[fit].any((-1,-2)).sum()),dev_pair_sources=int(valid[dev].any((-1,-2)).sum()),fit_target_combinations=int(valid[fit].sum()),dev_target_combinations=int(valid[dev].sum()),cache_sha256=sha(a.output/'cache.pt'));write(a.output/'summary.json',report);print(json.dumps(report))


if __name__=='__main__':main()
