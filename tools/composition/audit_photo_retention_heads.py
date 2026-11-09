#!/usr/bin/env python3
"""Replay selected retention heads and compare fourteen-family weak diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch
from torch import nn

from artifact_identity import sha
from analyze_unsplash_ranks import paired
from fit_photo_retention import ARMS,evaluate,vector
from fit_extents import retain


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=a.root/'features.pt'
    if sha(path)!='2b36c1215840dbc889481f6716eaddb5ba43d4b5af6fa434e3d1be6e91a9a3f1':
        raise ValueError('registered qualified feature cache differs')
    cache=torch.load(path,map_location='cpu',weights_only=False);sources=cache['sources']
    data={k:cache[k].cuda() for k in ['image','permuted','scores','eligible','bank_valid','contained','fraction']}
    data['ratios']=cache['ratios'];dev=[i for i,s in enumerate(sources) if s['role']=='dev']
    fit_progress=json.loads((a.root/'fits/progress.json').read_text())
    runs=json.loads((a.root/'fits/summary.json').read_text())
    if (fit_progress['pid'] is not None or fit_progress['completed_runs']!=9
        or {(r['arm'],r['seed']) for r in runs}!={(arm,seed) for arm in ARMS for seed in [31,137,911]}):
        raise ValueError('all nine completed controls required')
    family_values=defaultdict(lambda:defaultdict(list));audits=[]
    for run in runs:
        folder=a.root/'fits'/(run['arm']+'-seed'+str(run['seed']))
        if sha(folder/'best.pt')!=run['best_weights_sha256']:raise ValueError('selected head changed')
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        head=nn.Sequential(nn.Linear(762,96),nn.ReLU(),nn.Linear(96,2)).cuda()
        head.load_state_dict(state['head']);head.eval()
        metrics,rows=evaluate(head,data,dev,run['arm'],sources)
        saved=json.loads((folder/'best-predictions.json').read_text())
        if metrics!=run['best_dev'] or rows!=saved:raise ValueError('selected development replay differs')
        index=torch.tensor(dev,device='cuda')[:,None].expand(-1,11).reshape(-1)
        query=torch.arange(11,device='cuda')[None].expand(15,-1).reshape(-1)
        raw=head(vector(data,index,query,run['arm'])).reshape(15,11,128,2)
        valid=data['bank_valid'][dev]
        brier=(((raw[...,0].sigmoid()-data['contained'][dev].float()).square()*valid).sum(-1)/valid.sum(-1)).mean(-1).cpu().tolist()
        mae=(((raw[...,1].sigmoid()-data['fraction'][dev]).abs()*valid).sum(-1)/valid.sum(-1)).mean(-1).cpu().tolist()
        by_source=[]
        for j,i in enumerate(dev):
            chosen=[r for r in rows if r['id']==sources[i]['id']]
            value=dict(containment_brier=brier[j],fraction_mae=mae[j],
                offered_fraction=statistics.mean(r['offered'] for r in chosen),
                violation_per_request=statistics.mean(r['offered'] and not r['known_boxes_contained'] for r in chosen),
                successful_crop_per_request=statistics.mean(r['offered'] and r['known_boxes_contained'] for r in chosen))
            family_values[run['arm']][sources[i]['family_id']].append(value)
            by_source.append(dict(id=sources[i]['id'],family_id=sources[i]['family_id'],**value))
        audits.append(dict(arm=run['arm'],seed=run['seed'],selected_epoch=run['best_epoch'],
            selected_state_sha256=run['best_weights_sha256'],metrics=metrics,by_source=by_source,
            exact_saved_metrics_and_decisions=True))
    family={arm:{f:{k:statistics.mean(v[k] for v in values) for k in values[0]}
        for f,values in grouped.items()} for arm,grouped in family_values.items()}
    if any(len(v)!=14 for v in family.values()):raise ValueError('fourteen development families required')
    comparisons={arm:paired(family['image'],family[arm]) for arm in ['geometry','permuted']}
    result=dict(runs=audits,paired_image_minus_control=comparisons,development_sources=15,development_families=14,
        family_metrics=family,completed_fits=9,exact_selected_replays=9,
        mean_metrics={arm:{k:statistics.mean(r['best_dev'][k] for r in runs if r['arm']==arm)
            for k in ['containment_brier','fraction_mae']} for arm in ARMS},
        selection_data=True,confirmatory=False,model_promoted=False,face_coverage_certified=False,
        human_artistic_quality_measured=False,new_test_inference=False,paid_model_api_calls=0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    retain(a.root/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','family_metrics']},indent=2))


if __name__=='__main__':main()
