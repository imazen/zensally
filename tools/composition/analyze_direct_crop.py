#!/usr/bin/env python3
"""Audit direct-window controls, family intervals and CPU/GPU native geometry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from direct_crop import DirectCropHead,decode,geometry


def vectors(root,arm):
    by_seed=[]
    for seed in [31,137,911]:
        rows=json.loads((root/(arm+'-seed'+str(seed))/'best-predictions.json').read_text());groups=defaultdict(list)
        for row in rows:
            if row['essential'] and row['reference_available']:groups[row['family_id']].append(row)
        by_seed.append({f:np.array([np.mean([r[k] for r in values]) for k in ['edge_error','best2_match','top1_match','top1_cut','any_cut']]) for f,values in groups.items()})
    families=sorted(by_seed[0])
    if any(set(v)!=set(families) for v in by_seed):raise ValueError('seed families differ')
    return families,np.mean([[r[f] for f in families] for r in by_seed],axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=torch.load(a.root/'cache/cache.pt',map_location='cpu',weights_only=False)
    if sha(a.root/'cache/cache.pt')!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('fixed source cache differs')
    rows=json.loads((a.root/'fits/summary.json').read_text())
    if len(rows)!=9 or any(not r['native_replay_exact'] for r in rows):raise ValueError('nine exact selected native replays required')
    families,image=vectors(a.root/'fits','image');averages={};paired={}
    for arm in ['image','geometry-only','permuted-image']:
        names,matrix=vectors(a.root/'fits',arm)
        if names!=families:raise ValueError('control families differ')
        averages[arm]={key:float(matrix[:,j].mean()) for j,key in enumerate(['edge_error','best2_match','top1_match','top1_cut','any_cut'])}
        if arm!='image':
            delta=image-matrix;index=np.random.default_rng(31).integers(0,len(families),size=(5000,len(families)));sample=delta[index].mean(1)
            paired[arm]={key:dict(delta=float(delta[:,j].mean()),ci95=np.quantile(sample[:,j],[.025,.975]).tolist()) for j,key in enumerate(['edge_error','best2_match','top1_match','top1_cut','any_cut'])}
    normalized=(cache['visual']-cache['mean'])/cache['std'];sizes=[s['native_size'] for s in cache['sources']];cpu_context=geometry(sizes,cache['ratios']);gpu_context=geometry(sizes,cache['ratios'],device='cuda')
    checks=[]
    for row in rows:
        path=a.root/'fits'/(row['arm']+'-seed'+str(row['seed']))/'best.pt'
        if sha(path)!=row['weights_sha256']:raise ValueError('selected model identity differs')
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=DirectCropHead();model.load_state_dict(state);model.eval()
        visual=torch.zeros_like(normalized) if row['arm']=='geometry-only' else normalized[cache['donors']] if row['arm']=='permuted-image' else normalized
        with torch.no_grad():
            o,s,_=model(visual,cpu_context['query']);cpu_boxes,cpu_native=decode(o,s,cpu_context,integer=True)
            model.cuda();o,s,_=model(visual.cuda(),gpu_context['query']);gpu_boxes,gpu_native=decode(o,s,gpu_context,integer=True)
        edge=float(((cpu_boxes-gpu_boxes.cpu()).abs()*128).amax())
        if edge>3:raise ValueError('CPU/GPU native rectangles exceed unchanged three-grid edge gate')
        for native in [cpu_native,gpu_native.cpu()]:
            extent=native[...,2:]-native[...,:2]
            if not bool(((native[...,:2]>=0)&(native[...,2:]<=cpu_context['sizes'][:,None,None])&(extent>0)).all()):raise ValueError('invalid native bounds')
            ratio=cpu_context['pairs'];cross=extent[...,0]*ratio[None,:,None,1]-extent[...,1]*ratio[None,:,None,0]
            if not bool((cross==0).all()):raise ValueError('native target aspect differs')
        checks.append(dict(arm=row['arm'],seed=row['seed'],rectangles=72*22*2,max_edge_128_grid=edge,native_identical=bool(torch.equal(cpu_native,gpu_native.cpu()))))
    failed=torch.load(a.root/'fits-initial-progress-failure/image-seed31/weights-000.pt',map_location='cpu',weights_only=False)
    corrected=torch.load(a.root/'fits/image-seed31/weights-000.pt',map_location='cpu',weights_only=False)
    if any(not torch.equal(failed['replay'][key],corrected['replay'][key]) for key in failed['replay']):raise ValueError('progress repair changed epoch-zero native outputs')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(averages=averages,paired=paired,families=len(families),cpu_gpu_native_checks=checks,progress_repair_initial_outputs_exact=True,
        bootstrap=dict(replicates=5000,seed=31,unit='source family',seeds='mean of3 before bootstrap'),
        old_quality_gate_measured=False,model_promoted=False,face_coverage_certified=False,api_calls=0,test_inference=False,
        limitation='Editorial-only direct-window suitability on reused DEV families and finite unreviewed Sol nominations; best-of-two is not a top1 quality metric.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
