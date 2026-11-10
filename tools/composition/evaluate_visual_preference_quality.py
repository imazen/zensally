#!/usr/bin/env python3
"""Apply the unchanged original full-model CUDA gate to nine review controls."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from artifact_identity import sha
from distill_attention_linear import Student, write
from evaluate_attention_fallback import StudentScorer
from fit_unsplash_ranks import evaluate, LocalCropRanker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','cache','weights','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('original inputs differ')
    for name in ['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):raise ValueError('frozen quality implementation differs')
    fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if {(r['arm'],r['seed']) for r in fits}!={(arm,seed) for arm in ['replay','review','reversed-review'] for seed in [31,137,911]} or len(fits)!=9:raise ValueError('registered nine controls required')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);sources=[dict(s,ratios=c['ratios']) for s in c['sources']];dev=[i for i,s in enumerate(sources) if s['role']=='dev']
    if len(dev)!=25:raise ValueError('original25 DEV sources required')
    data={k:c[k].cuda() for k in ['image','content','boxes','eligible','targets','pairs','pair_valid','query_valid']};data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda');data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in c['ratios'] for _ in range(2)],device='cuda');data['mode']=torch.tensor([q%2 for q in range(22)],device='cuda',dtype=torch.float32);donors=torch.arange(len(sources),device='cuda');initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];base=LocalCropRanker().cuda();base.load_state_dict(initial)
    baseline,base_rows=evaluate(base,data,dev,donors,sources);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'baseline-predictions.json',base_rows);result=[]
    for fit in fits:
        name=f"{fit['arm']}-seed{fit['seed']}";path=a.fits/name/'selected.pt';s=torch.load(path,map_location='cpu',weights_only=False)['model'];model=LocalCropRanker().cuda();model.load_state_dict(initial);student=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student.load_state_dict(s);model.ranker.scorer[2]=StudentScorer(student).cuda();metrics,rows=evaluate(model,data,dev,donors,sources);write(a.output/(name+'-predictions.json'),rows);result.append(dict(arm=fit['arm'],seed=fit['seed'],selected_epoch=fit['selected_epoch'],checkpoint_sha256=sha(path),metrics=metrics,old_essential_nonregression=metrics['essential']['teacher_regret']<=baseline['essential']['teacher_regret']))
    write(a.output/'summary.json',dict(baseline=baseline,rows=result,api_calls=0,test_inference=False,model_promoted=False,face_coverage_certified=False));print(json.dumps([dict(arm=r['arm'],seed=r['seed'],regret=r['metrics']['essential']['teacher_regret'],passes=r['old_essential_nonregression']) for r in result]))


if __name__=='__main__':main()
