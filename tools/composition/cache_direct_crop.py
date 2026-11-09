#!/usr/bin/env python3
"""Cache frozen source-grid features and existing Sol weak windows, no new labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker
from crop_regions import area_pool


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','references','initializer','output']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.references)!='f639eb3a3682f6c4a6f6511346f212d3fd05e5076c360be90610bd93c4900199' or sha(a.initializer)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('frozen Sol/C002 inputs differ')
    parent=torch.load(a.references,map_location='cpu',weights_only=False)
    if Counter(s['role'] for s in parent['sources'])!=dict(fit=48,dev=24):raise ValueError('fixed48/24 roles required')
    if {s['family_id'] for s in parent['sources'] if s['role']=='fit'}&{s['family_id'] for s in parent['sources'] if s['role']=='dev'}:raise ValueError('family crosses role')
    model=LocalCropRanker().cuda();model.load_state_dict(torch.load(a.initializer,map_location='cpu',weights_only=False)['model']);model.eval().requires_grad_(False)
    grid=torch.tensor([[x/8,y/8,(x+1)/8,(y+1)/8] for y in range(8) for x in range(8)],device='cuda')
    content=parent['content'].cuda();offset=content[:,:2].repeat(1,2)[:,None];scale=(content[:,2:]-content[:,:2]).repeat(1,2)[:,None]
    rectangles=offset+grid[None]*scale
    with torch.inference_mode():
        features=encode(model,parent['image'].cuda(),list(range(72)))
        visual=torch.cat([area_pool(value,rectangles).flatten(1) for value in features],-1).cpu()
    if visual.shape!=(72,2048) or not bool(torch.isfinite(visual).all()):raise ValueError('finite72x2048 source features required')
    offer=torch.zeros((72,22));offer_valid=torch.zeros((72,22),dtype=torch.bool);rois=[]
    for i,source in enumerate(parent['sources']):
        heads=[]
        if source['annotation_valid']:
            zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text())
            if zones['purpose']!='editorial':raise ValueError('this scout admits editorial only')
            heads=[r['box'] for r in zones['rois'] if r['kind']=='face_head']
            for request in zones['crops']:
                q=parent['ratios'].index(request['ratio'])*2+(request['mode']=='maximal')
                known=bool(parent['reference_valid'][i,q].any());offer[i,q]=known;offer_valid[i,q]=known or request['action']=='contain'
        rois.append(heads)
    fit=[i for i,s in enumerate(parent['sources']) if s['role']=='fit'];mean=visual[fit].mean(0);std=visual[fit].std(0,correction=0).clamp_min(.001)
    groups={};donors=list(range(72))
    for i,source in enumerate(parent['sources']):groups.setdefault((source['role'],tuple((parent['content'][i]*256).round().int().tolist())),[]).append(i)
    for values in groups.values():
        ordered=sorted(values,key=lambda i:hashlib.sha256(('direct-crop-permutation|'+parent['sources'][i]['id']).encode()).hexdigest())
        for i,j in zip(ordered,ordered[1:]+ordered[:1]):donors[i]=j
    result=dict(schema='direct-crop-source-grid-v1',sources=parent['sources'],ratios=parent['ratios'],visual=visual,mean=mean,std=std,donors=donors,
        references=parent['references'],reference_valid=parent['reference_valid'],offer=offer,offer_valid=offer_valid,head_rois=rois,
        protocol=dict(parent_references_sha256=sha(a.references),encoder_sha256=sha(a.initializer),input_size=256,feature_grid=8,channels_per_level=16,
            feature_normalization_fit_only=True,purpose='editorial',labels='finite unreviewed Sol nominations and explicit containment recommendations',
            api_calls=0,source_roles_changed=False,test_inference=False,synthetic_gradients=False,face_coverage_certified=False,model_promoted=False))
    a.output.mkdir(parents=True,exist_ok=False);torch.save(result,a.output/'cache.pt')
    summary=dict(sources=72,fit=48,dev=24,features=2048,cache_sha256=sha(a.output/'cache.pt'),cache_bytes=(a.output/'cache.pt').stat().st_size,
        accepted_queries=int(parent['reference_valid'].any(-1).sum()),offer_known=int(offer_valid.sum()),explicit_contain=int((offer_valid&~offer.bool()).sum()),
        permuted_sources=sum(i!=j for i,j in enumerate(donors)),cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
