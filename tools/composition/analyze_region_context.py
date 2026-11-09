#!/usr/bin/env python3
"""Audit matched region-feature fits, repeated controls and native CPU/GPU geometry."""
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import torch
from artifact_identity import sha
from analyze_direct_crop import vectors
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from direct_crop import DirectCropHead,decode,geometry

VARIANTS=['none','local-face','local-pose','local-face-pose','oracle-heads']


def offer_vectors(root,arm):
    seed_rows=[]
    for seed in [31,137,911]:
        groups=defaultdict(list)
        for row in json.loads((root/f'{arm}-seed{seed}'/'best-predictions.json').read_text()):
            if row['essential'] and row['offer_known']:groups[row['family_id']].append(row)
        seed_rows.append({f:np.mean([[r[k] for k in ['brier','offered','false_offer','missed_offer']] for r in rows],axis=0) for f,rows in groups.items()})
    families=sorted(seed_rows[0])
    if any(set(v)!=set(families) for v in seed_rows):raise ValueError('offer families differ by seed')
    return families,np.mean([[v[f] for f in families] for v in seed_rows],axis=0)


def paired(left,right):
    delta=left-right;indices=np.random.default_rng(31).integers(0,len(delta),size=(5000,len(delta)));sample=delta[indices].mean(1)
    return [dict(delta=float(delta[:,j].mean()),ci95=np.quantile(sample[:,j],[.025,.975]).tolist()) for j in range(delta.shape[1])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','direct','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.root/'cache/cache.pt')!='e79024b7d887e19633d8624dddeb8ecde2f2dd8a1ffd7cd8b704eb5391684b30' or sha(a.direct/'cache/cache.pt')!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('fixed region/direct cache differs')
    regions=torch.load(a.root/'cache/cache.pt',map_location='cpu',weights_only=False);cache=torch.load(a.direct/'cache/cache.pt',map_location='cpu',weights_only=False)
    base=(cache['visual']-cache['mean'])/cache['std'];fit=[i for i,s in enumerate(cache['sources']) if s['role']=='fit']
    sizes=[s['native_size'] for s in cache['sources']];cpu_context=geometry(sizes,cache['ratios']);gpu_context=geometry(sizes,cache['ratios'],device='cuda')
    metrics={};values={};checks=[];controls={};identities={}
    for variant in VARIANTS:
        root=a.root/variant/'fits';progress=json.loads((root/'PROGRESS.json').read_text());reports=json.loads((root/'summary.json').read_text())
        if progress['stage']!='complete' or progress['completed']!=9 or progress['pid'] is not None or len(reports)!=9:raise ValueError('all nine fits required for each variant')
        extra=regions['variants'][variant];normalized=torch.cat((base,(extra-extra[fit].mean(0))/extra[fit].std(0,correction=0).clamp_min(.001)),-1)
        metrics[variant]={};values[variant]={};identities[variant]=[]
        for arm in ['image','geometry-only','permuted-image']:
            families,matrix=vectors(root,arm);offer_families,offer_matrix=offer_vectors(root,arm)
            metrics[variant][arm]=dict(reference_families=len(families),offer_families=len(offer_families),
                **dict(zip(['edge_error','best2_match','top1_match','top1_cut','any_cut'],map(float,matrix.mean(0)))),
                **dict(zip(['brier','offered','false_offer','missed_offer'],map(float,offer_matrix.mean(0)))))
            values[variant][arm]=(families,matrix,offer_families,offer_matrix)
        for record in reports:
            path=root/(record['arm']+'-seed'+str(record['seed']))/'best.pt'
            if not record['native_replay_exact'] or sha(path)!=record['weights_sha256']:raise ValueError('selected identity/native replay differs')
            state=torch.load(path,map_location='cpu',weights_only=False)['model']
            if record['arm']=='geometry-only':
                if variant=='none':controls[record['seed']]=state
                elif any(not torch.equal(v,controls[record['seed']][k]) for k,v in state.items()):raise ValueError('repeated geometry controls differ across region variants')
            model=DirectCropHead(visual=2095);model.load_state_dict(state);model.eval()
            visual=torch.zeros_like(normalized) if record['arm']=='geometry-only' else normalized[cache['donors']] if record['arm']=='permuted-image' else normalized
            with torch.no_grad():
                o,s,_=model(visual,cpu_context['query']);cpu_boxes,cpu_native=decode(o,s,cpu_context,integer=True)
                model.cuda();o,s,_=model(visual.cuda(),gpu_context['query']);gpu_boxes,gpu_native=decode(o,s,gpu_context,integer=True)
            motion=float(((cpu_boxes-gpu_boxes.cpu()).abs()*128).amax())
            if motion>3:raise ValueError('native CPU/GPU rectangles exceed unchanged three-grid edge gate')
            checks.append(dict(variant=variant,arm=record['arm'],seed=record['seed'],rectangles=3168,max_edge_128_grid=motion,native_identical=bool(torch.equal(cpu_native,gpu_native.cpu()))))
            identities[variant].append(dict(arm=record['arm'],seed=record['seed'],weights_sha256=record['weights_sha256'],epoch=record['epoch']))
    comparisons={}
    for variant in VARIANTS[1:]:
        comparisons[variant]={}
        for label,reference in [('matched-none',values['none']['image']),('same-variant-geometry',values[variant]['geometry-only']),('same-variant-permuted',values[variant]['permuted-image'])]:
            current=values[variant]['image']
            if current[0]!=reference[0] or current[2]!=reference[2]:raise ValueError('paired comparison family sets differ')
            comparisons[variant][label]=dict(zip(['edge_error','best2_match','top1_match','top1_cut','any_cut'],paired(current[1],reference[1])))
            comparisons[variant][label].update(dict(zip(['brier','offered','false_offer','missed_offer'],paired(current[3],reference[3]))))
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(averages=metrics,paired=comparisons,identities=identities,cpu_gpu_native_checks=checks,repeated_geometry_controls_bit_identical=True,
        completed_fits=45,all_selected_native_replays_exact=True,bootstrap=dict(replicates=5000,seed=31,unit='source family',seeds='mean of3 before bootstrap'),
        model_promoted=False,old_quality_gate_measured=False,face_coverage_certified=False,api_calls=0,test_inference=False,
        limitation='Reused editorial DEV families and finite unreviewed windows; the oracle descriptor is a bounded same-call region input, not a deployment localization result.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(image_averages={k:v['image'] for k,v in metrics.items()},paired_to_none={k:v['matched-none'] for k,v in comparisons.items()},repeated_geometry_controls_bit_identical=True)))


if __name__=='__main__':main()
