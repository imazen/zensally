#!/usr/bin/env python3
"""Audit exact per-person origin projection of fixed direct-head predictions."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import torch
import numpy as np
from artifact_identity import sha
from subject_origin_projection import nearest
from subject_policy import face_relation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from direct_crop import DirectCropHead,decode,geometry
from fit_direct_crop import subset
from subject_geometry import pixels


@lru_cache(maxsize=512)
def projected(w,h,raw,heads,ratio,mode,broad):
    result=nearest(w,h,list(raw),heads);fallback=False
    if result is None and mode=='maximal' and broad:
        p,q=map(int,ratio.split(':'));g=math.gcd(p,q);p,q=p//g,q//g;k=min(w//p,h//q);cw,ch=p*k,q*k
        x=max(0,min(w-cw,(raw[0]+raw[2]-cw)//2));y=max(0,min(h-ch,(raw[1]+raw[3]-ch)//2))
        result=nearest(w,h,[x,y,x+cw,y+ch],heads);fallback=result is not None
    return tuple(result) if result is not None else None,fallback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','direct','regions','cues','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.direct/'cache/cache.pt')!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472' or sha(a.regions/'cache/cache.pt')!='e79024b7d887e19633d8624dddeb8ecde2f2dd8a1ffd7cd8b704eb5391684b30':raise ValueError('fixed source parents differ')
    cache=torch.load(a.direct/'cache/cache.pt',map_location='cpu',weights_only=False)
    normalized=torch.cat(((cache['visual']-cache['mean'])/cache['std'],torch.zeros(72,47)),-1)
    sizes=[s['native_size'] for s in cache['sources']];context=geometry(sizes,cache['ratios']);gpu_context=geometry(sizes,cache['ratios'],device='cuda')
    local={}
    for row in json.loads((a.cues/'analysis/boxes.json').read_text()):local.setdefault(row['id'],{})[row['model']]=row['boxes']
    paths={};native={};offers={};raw_checks=[]
    reports=json.loads((a.regions/'none/fits/summary.json').read_text())
    for seed in [31,137,911]:
        row=next(r for r in reports if r['arm']=='image' and r['seed']==seed);path=a.regions/'none/fits'/f'image-seed{seed}'/'best.pt'
        if sha(path)!=row['weights_sha256'] or not row['native_replay_exact']:raise ValueError('fixed direct-head identity differs')
        paths[str(seed)]=sha(path);state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=DirectCropHead(visual=2095);model.load_state_dict(state);model.eval()
        with torch.no_grad():
            o,s,offer=model(normalized,context['query']);_,batch=decode(o,s,context,integer=True);single=[];single_offers=[]
            for i in range(72):
                index=torch.tensor([i]);o,s,v=model(normalized[index],context['query'][index]);_,one=decode(o,s,subset(context,index),integer=True);single.append(one);single_offers.append(v)
            model.cuda();o,s,gpu_offer=model(normalized.cuda(),gpu_context['query']);_,gpu=decode(o,s,gpu_context,integer=True)
        native[seed]={'cpu-batch':batch.int().tolist(),'cpu-single':torch.cat(single).int().tolist(),'gpu-batch':gpu.cpu().int().tolist()}
        offers[seed]={'cpu-batch':(offer>=0).tolist(),'cpu-single':(torch.cat(single_offers)>=0).tolist(),'gpu-batch':(gpu_offer>=0).cpu().tolist()}
        max_edge=max(float(((batch-other.cpu()).abs()/context['sizes'][:,None,None].repeat(1,1,1,2)*128).amax()) for other in [torch.cat(single),gpu])
        raw_checks.append(dict(seed=seed,max_edge_128_grid=max_edge,raw_gate_passed=max_edge<=3))
        del model
    a.output.mkdir(parents=True,exist_ok=False);rows=[];failures=[]
    protocol=dict(weights_sha256=paths,source_cache_sha256=sha(a.direct/'cache/cache.pt'),region_cache_sha256=sha(a.regions/'cache/cache.pt'),
        models='three preselected image/no-region controls; no re-selection for projection results',
        policies=['nearest fixed extent','nearest fixed extent with largest Maximal extent only when fixed extent is infeasible'],
        constraint='at least one supplied head wholly contained; every other supplied head wholly contained or excluded',
        objective='minimum squared source-normalized origin movement; integer comparison; deterministic y/x ties',
        numerical_gate='CPU batch, CPU single-image and GPU batch: same crop presence/options and at most3 source-normalized128-grid pixels per edge',
        api_calls=0,model_promoted=False,test_inference=False,face_coverage_certified=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for i,source in enumerate(cache['sources']):
        (a.output/'PROGRESS.json').write_text(json.dumps(dict(pid=os.getpid(),completed_sources=i,required_sources=72,stage='projecting'))+'\n')
        if not source['annotation_valid']:continue
        w,h=source['native_size'];truth=[pixels(r,w,h,outward=True) for r in cache['head_rois'][i]]
        zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text());requested={(r['ratio'],r['mode']) for r in zones['crops']}
        face=local[source['id']]['float16-blaze_face_full_range_sparse'];pose=local[source['id']]['float16-pose_landmarker_lite']
        regions=dict(provided=truth,local_face=[[math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)] for b in face],
            local_face_pose=[[math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)] for b in face+pose])
        for q in range(22):
            ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal'
            if (ratio,mode) not in requested:continue
            references=cache['references'][i,q][cache['reference_valid'][i,q]].numpy();scale=np.array([w,h,w,h])
            for seed in [31,137,911]:
                for kind,heads in regions.items():
                    heads=tuple(tuple(b) for b in heads)
                    for broad in [False,True]:
                        versions={};fallbacks={}
                        for version,value in native[seed].items():
                            result=[];fallback=False
                            for raw in value[i][q]:
                                box,f=projected(w,h,tuple(raw),heads,ratio,mode,broad);fallback|=f
                                if box is not None and box not in result:result.append(box)
                            versions[version]=result;fallbacks[version]=fallback
                        reference=versions['cpu-batch']
                        for version,result in versions.items():
                            same_presence=len(reference)==len(result);motion=max((float((np.abs(np.array(a)-b)/scale*128).max()) for a,b in zip(reference,result)),default=0.)
                            same_offer=offers[seed]['cpu-batch'][i][q]==offers[seed][version][i][q]
                            if not same_presence or motion>3 or not same_offer:
                                failures.append(dict(id=source['id'],seed=seed,query=q,regions=kind,broad=broad,version=version,reference=reference,candidate=result,
                                    max_edge_128_grid=motion,same_presence=same_presence,same_model_offer=same_offer))
                        cut=any(any(face_relation(list(box),head)=='cut' for head in truth) for box in reference)
                        if kind=='provided' and cut:raise ValueError('exact projection cut a supplied head')
                        match=any(float((np.abs(np.asarray(box)/scale-target)*128).max())<=3 for box in reference for target in references)
                        for use_offer in [False,True]:
                            offered=bool(reference) and (not use_offer or offers[seed]['cpu-batch'][i][q])
                            rows.append(dict(id=source['id'],role=source['role'],family_id=source['family_id'],seed=seed,query=q,ratio=ratio,mode=mode,essential=q<18,
                                regions=kind,broad_fallback=broad,use_model_offer=use_offer,offered=offered,cut=offered and cut,
                                reference_available=bool(len(references)),reference_match=offered and match,native_boxes=reference if offered else [],
                                largest_extent_used=offered and fallbacks['cpu-batch']))
    summary=[]
    for kind in ['provided','local_face','local_face_pose']:
        for broad in [False,True]:
            for use_offer in [False,True]:
                for scope in ['all-requested','accepted-references']:
                    chosen=[r for r in rows if r['role']=='dev' and r['essential'] and r['regions']==kind and r['broad_fallback']==broad and r['use_model_offer']==use_offer and (scope=='all-requested' or r['reference_available'])]
                    groups=defaultdict(list)
                    for row in chosen:groups[row['family_id']].append(row)
                    summary.append(dict(regions=kind,broad_fallback=broad,use_model_offer=use_offer,scope=scope,families=len(groups),requests=len(chosen)//3,
                        **{key:float(np.mean([np.mean([r[key] for r in values]) for values in groups.values()])) for key in ['offered','cut','reference_match','largest_extent_used']}))
    result=dict(summary=summary,raw_numerical_checks=raw_checks,numerical_failures=failures,numerical_gate_passed=not failures and all(r['raw_gate_passed'] for r in raw_checks),
        cache_info=projected.cache_info()._asdict(),model_promoted=False,face_coverage_certified=False,api_calls=0,test_inference=False,
        limitation='Nearest legal origin for supplied regions is exact; local regions and Sol nominations do not certify actual-face completeness. Reused DEV matching is not artistic gold.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'predictions.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    (a.output/'PROGRESS.json').write_text(json.dumps(dict(pid=None,completed_sources=72,required_sources=72,stage='complete' if result['numerical_gate_passed'] else 'complete-with-numerical-failures'))+'\n')
    print(json.dumps(dict(summary=[r for r in summary if r['scope']=='all-requested' and not r['use_model_offer']],numerical_failures=len(failures),numerical_gate_passed=result['numerical_gate_passed'])))
    if not result['numerical_gate_passed']:raise SystemExit(1)


if __name__=='__main__':main()
