#!/usr/bin/env python3
"""Full CUDA original quality and cached FIT/DEV pair diagnostics, no TEST."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from artifact_identity import sha
from distill_attention_linear import Student,write
from evaluate_attention_fallback import StudentScorer
from fit_unsplash_ranks import evaluate,encode,LocalCropRanker
from fit_nested_pairs import paired_metrics,ARMS,BASELINE
from nested_pair_training import compatibility,pair_choices
from nested_pair_selection import select
from local_rank_distillation import family_weights


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','hidden','cache','new','weights','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected=[(a.hidden,'41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'),(a.cache,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('frozen inputs differ')
    for name in ['training/local_rank_distillation.py','training/crop_regions.py','training/crop_ranker.py','training/model_v3ds.py','tools/composition/fit_unsplash_ranks.py']:
        if Path(name).read_bytes()!=subprocess.check_output(['git','show','9f12692417d774653ed676d236e0f89eaafa7385:'+name]):raise ValueError('frozen quality implementation differs')
    fits=json.loads((a.fits/'summary.json').read_text())['fits']
    if len(fits)!=9 or {(r['arm'],r['seed']) for r in fits}!={(arm,seed) for arm in ARMS for seed in [31,137,911]}:raise ValueError('registered nine fits required')
    old=torch.load(a.hidden,map_location='cpu',weights_only=False);c=torch.load(a.cache,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);sources=[dict(s,ratios=c['ratios']) for s in c['sources']];dev=[i for i,s in enumerate(sources) if s['role']=='dev'];fit=[i for i,s in enumerate(sources) if s['role']=='fit']
    if len(dev)!=25 or len(fit)!=94 or any(s['role'] not in ['fit','dev'] for s in sources+new['sources']):raise ValueError('fixed FIT/DEV only')
    scale=torch.tensor([[*s['native_size'],*s['native_size']] for s in sources])[:,None,None,:];boxes=(c['boxes']*scale).round().long();mask=compatibility(boxes,c['eligible'].reshape(119,11,2,128));labels=pair_choices(c['targets'].reshape(119,11,2,128),mask)
    data={k:c[k].cuda() for k in ['image','content','boxes','eligible','targets','pairs','pair_valid','query_valid']};data['source_aspect']=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources],device='cuda');data['aspect']=torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in c['ratios'] for _ in range(2)],device='cuda');data['mode']=torch.tensor([q%2 for q in range(22)],device='cuda',dtype=torch.float32);donors=torch.arange(119,device='cuda');initial=torch.load(a.weights,map_location='cpu',weights_only=False)['model'];base=LocalCropRanker().cuda();base.load_state_dict(initial);baseline,base_rows=evaluate(base,data,dev,donors,sources)
    if baseline['essential']['teacher_regret']!=BASELINE:raise ValueError('original full-model baseline changed')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'baseline-predictions.json',base_rows);result=[];numerical=[]
    for entry in fits:
        name=f"{entry['arm']}-seed{entry['seed']}";path=a.fits/name/('selected.pt' if entry['qualified'] else 'best-ineligible.pt');state=torch.load(path,map_location='cpu',weights_only=False);s=state['model'];student=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);student.load_state_dict(s);student.eval();cached=student(old['hidden'],None);cohorts=[]
        for role,indices in [('fit',fit),('dev',dev)]:
            regret,choices=paired_metrics(cached[indices],old,mask,indices);valid=c['query_valid'][indices].reshape(len(indices),11,2).all(-1);agreement=(choices.reshape(len(indices),11,2)==labels[indices]).all(-1)
            cohorts.append(dict(role=role,paired_essential_regret=regret,both_valid_pairs=int(valid.sum()),exact_target_sum_pair_matches=int((agreement&valid).sum())))
        compact=student.compact_scores(old['hidden']);maxdiff=float((compact-cached).abs().max());changed=0;gate=0;maxedge=0.;selector_disagreements=0
        for scoreset in [cached,compact]:
            choices=pair_choices(scoreset.reshape(119,11,2,128),mask)
            if scoreset is cached:reference=choices
            else:
                changed=int((choices!=reference).sum())
                for i in range(119):
                    for ri in range(11):
                        b=boxes[i,ri];motion=(b[choices[i,ri]]-b[reference[i,ri]]).abs().float()/scale[i,0,0]*128;edge=float(motion.max());maxedge=max(maxedge,edge);gate+=edge>3
            if scoreset is cached:
                for i in dev:
                    for ri in range(11):
                        geometry=boxes[i,ri].tolist();q=ri*2;oracle=select(geometry,geometry,*scoreset[i,q:q+2].tolist(),*c['eligible'][i,q:q+2].tolist(),'joint-sum');selector_disagreements+=tuple(choices[i,ri].tolist())!=oracle
        # New data has per-mode banks; retain raw compact arithmetic separately.
        nc=student(new['hidden'],None);nf=student.compact_scores(new['hidden']);ne=new['eligible'];new_changes=int((nc.masked_fill(~ne,-torch.inf).argmax(-1)!=nf.masked_fill(~ne,-torch.inf).argmax(-1)).sum());maxdiff=max(maxdiff,float((nc-nf).abs().max()))
        model=LocalCropRanker().cuda();model.load_state_dict(initial);model.ranker.scorer[2]=StudentScorer(student).cuda();metrics,rows=evaluate(model,data,dev,donors,sources);write(a.output/(name+'-predictions.json'),rows)
        features=encode(model,data['image'],donors[dev]);full=[]
        for q in range(22):
            full.append(model.scores(features,data['boxes'][dev,q//2],data['source_aspect'][dev],data['aspect'][q].expand(len(dev)),data['mode'][q].expand(len(dev)),data['content'][dev]).cpu())
        full=torch.stack(full,1);full_regret,full_choices=paired_metrics(full,old,mask,dev);cpu_choices=pair_choices(cached[dev].reshape(25,11,2,128),mask[dev]).reshape(25,22);cuda_changes=int((full_choices!=cpu_choices).sum());cuda_maxedge=0.;cuda_gate=0
        for j,i in enumerate(dev):
            for q in range(22):
                edge=float(((boxes[i,q//2,full_choices[j,q]]-boxes[i,q//2,cpu_choices[j,q]]).abs().float()/scale[i,0,0]*128).max());cuda_maxedge=max(cuda_maxedge,edge);cuda_gate+=edge>3
        numerical.append(dict(arm=entry['arm'],seed=entry['seed'],old_pair_requests=1309,compact_old_endpoint_changes=changed,compact_old_pair_gate_violations=gate,compact_old_max_edge_grid=maxedge,compact_new_raw_endpoint_changes=new_changes,max_absolute_compact_difference=maxdiff,python_selector_disagreements=selector_disagreements,cuda_pair_endpoint_changes=cuda_changes,cuda_pair_gate_violations=cuda_gate,cuda_pair_max_edge_grid=cuda_maxedge))
        result.append(dict(arm=entry['arm'],seed=entry['seed'],epoch=entry['epoch'],qualified=entry['qualified'],checkpoint_sha256=sha(path),cached_cohorts=cohorts,full_raw_regret=metrics['essential']['teacher_regret'],full_paired_regret=full_regret,full_both_quality_gates=metrics['essential']['teacher_regret']<=BASELINE and full_regret<=BASELINE))
    write(a.output/'summary.json',dict(baseline=baseline,rows=result,numerical=numerical,api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(result));print(json.dumps(numerical))
    if any(r['compact_old_pair_gate_violations'] or r['cuda_pair_gate_violations'] or r['python_selector_disagreements'] or r['compact_new_raw_endpoint_changes'] for r in numerical):raise SystemExit('numerical gate failed')


if __name__=='__main__':main()
