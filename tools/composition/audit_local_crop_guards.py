#!/usr/bin/env python3
"""Measure fixed crop heads constrained by local proposals versus supplied heads."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from artifact_identity import sha
from fit_unsplash_ranks import encode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from local_rank_distillation import LocalCropRanker
from subject_geometry import pixels
from subject_policy import face_relation


def legal_mask(bank,regions,eligible):
    states=[[face_relation(box,region) for region in regions] for box in bank]
    return eligible&np.array(['contained' in s and 'cut' not in s for s in states])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','cache','fits','cues','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(a.cache)!='bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9':raise ValueError('fixed Sol hybrid cache differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);dev=[i for i,s in enumerate(cache['sources']) if s['role']=='dev']
    local={}
    for row in json.loads((a.cues/'analysis/boxes.json').read_text()):local.setdefault(row['id'],{})[row['model']]=row['boxes']
    image=cache['image'][dev].cuda();content=cache['content'][dev].cuda();boxes=cache['query_boxes'][dev].cuda()
    aspect=torch.tensor([cache['sources'][i]['native_size'][0]/cache['sources'][i]['native_size'][1] for i in dev],device='cuda')
    reports=json.loads((a.fits/'summary.json').read_text());scores={};features=None;weights={}
    for seed in [31,137,911]:
        report=next(r for r in reports if r['arm']=='frozen-image' and r['seed']==seed);path=a.fits/f'frozen-image-seed{seed}'/'best.pt'
        if not report['old_nonregression'] or not report['replay_exact'] or sha(path)!=report['weights_sha256']:raise ValueError('fixed head quality/identity differs')
        model=LocalCropRanker().cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval();matrices=[]
        with torch.no_grad():
            if features is None:features=encode(model,image,list(range(len(dev))))
            for q in range(22):
                p,r=map(int,cache['ratios'][q//2].split(':'));value=model.scores(features,boxes[:,q],aspect,aspect.new_full((len(dev),),p/r),aspect.new_full((len(dev),),q%2),content)
                if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite crop scores')
                matrices.append(value.cpu().numpy())
        scores[seed]=matrices;weights[str(seed)]=sha(path);del model
    rows=[]
    for j,index in enumerate(dev):
        source=cache['sources'][index]
        if not source['annotation_valid']:continue
        zones=json.loads((a.run/'vision-cache'/source['reference_cache_key']/'zones.json').read_text());w,h=source['native_size']
        heads=[pixels(r['box'],w,h,outward=True) for r in zones['rois'] if r['kind']=='face_head']
        requests={(r['ratio'],r['mode']) for r in zones['crops']}
        regions={name:[[math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)] for b in value] for name,value in local[source['id']].items()}
        regions['provided-heads']=heads
        for q in range(22):
            ratio=cache['ratios'][q//2];mode='minimal' if q%2==0 else 'maximal'
            if (ratio,mode) not in requests:continue
            bank=np.rint(cache['query_boxes'][index,q].numpy()*np.array([w,h,w,h])).astype(int).tolist();eligible=cache['eligible'][index,q].numpy()
            masks=dict(raw=eligible,**{name:legal_mask(bank,value,eligible) for name,value in regions.items()})
            for policy,mask in masks.items():
                for seed,matrices in scores.items():
                    choice=int(np.argmax(np.where(mask,matrices[q][j],-np.inf))) if bool(mask.any()) else None
                    rows.append(dict(policy=policy,seed=seed,id=source['id'],family_id=source['family_id'],query=q,ratio=ratio,mode=mode,essential=q<18,
                        offered=choice is not None,selected=choice,native_box=bank[choice] if choice is not None else None,
                        nominated_head_cut=bool(choice is not None and cache['negative'][index,q,choice]),
                        reference_available=bool(cache['reference_valid'][index,q].any()),reference_match=bool(choice is not None and cache['positive'][index,q,choice])))
    summary=[]
    for policy in sorted({r['policy'] for r in rows}):
        for scope in ['all-requested','accepted-references']:
            chosen=[r for r in rows if r['policy']==policy and r['essential'] and (scope=='all-requested' or r['reference_available'])];groups=defaultdict(list)
            for row in chosen:groups[row['family_id']].append(row)
            summary.append(dict(policy=policy,scope=scope,families=len(groups),requests=len(chosen)//3,
                **{key:float(np.mean([np.mean([r[key] for r in group]) for group in groups.values()])) for key in ['offered','nominated_head_cut','reference_match']}))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'predictions.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    result=dict(summary=summary,weights_sha256=weights,cache_sha256=sha(a.cache),
        face_coverage_certified=False,model_promoted=False,test_inference=False,api_calls=0,
        limitation='Each local guard preserves one returned region and excludes partial returned regions; missing actual faces remain possible. Sol head proposals are an incomplete annotation-scope diagnostic.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
