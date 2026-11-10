#!/usr/bin/env python3
"""Fixed-model retention evaluation using cached weak references, no new labels."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

from artifact_identity import sha
from analyze_unsplash_ranks import paired
from constrained_bank import candidates
from fit_extents import retain
from fit_photo_retention import evaluate,vector,ARMS
from region_retention import retained_fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker

RATIOS=['1:1','4:3','3:4','16:9','9:16','2:1','1:2','3:1','1:3','6:1','1:6']


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','references','weights']:p.add_argument(name,type=Path)
    a=p.parse_args()
    plan=json.loads((a.root/'plan.json').read_text());sources=plan['sources'];donors=plan['source_permutation']
    if len(sources)!=15 or len(plan['fixed_models'])!=9 or plan['offer_threshold']!=.5:raise ValueError('frozen fifteen-source/nine-model protocol required')
    if sha(a.references/'SHA256SUMS.json')!=plan['reference_packet_sha256']:raise ValueError('frozen reference packet differs')
    if sha(a.weights)!='c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':raise ValueError('original photographic ranker differs')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    progress=a.root/'progress.json'
    progress.write_text(json.dumps(dict(pid=os.getpid(),stage='inputs',required_models=9,completed_models=0,checked_unix=time.time()))+'\n')
    calls={r['id']:r for r in json.loads((a.references/'references/supplementary15-summary.json').read_text())['calls']}
    images,content,aspects,all_boxes,valids,contained,fractions,eligible,receipts,native_banks=[],[],[],[],[],[],[],[],[],[]
    for i,s in enumerate(sources):
        if s['role']!='supplementary_dev' or s['fit_eligible'] or s['valid_rect']!=sources[donors[i]]['valid_rect']:raise ValueError('source role/donor shape differs')
        raw=Path(s['tensor_path'])
        if sha(raw)!=s['tensor_sha256'] or raw.stat().st_size!=256*256*3:raise ValueError('owned neural input differs')
        images.append(torch.from_numpy(np.frombuffer(raw.read_bytes(),np.uint8).copy().reshape(256,256,3)).permute(2,0,1).float()/255)
        content.append(s['valid_rect']);w,h=s['native_size'];aspects.append(w/h)
        call=calls[s['id']];path=a.references/'references/responses'/call['response'];row=json.loads(path.read_text());contract=row['contract']
        digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (digest!=path.stem or digest!=row['contract_sha256'] or call['invalid']
            or contract['source_id']!=s['id'] or contract['source_role']!='supplementary_dev'
            or contract['source_plan_sha256']!=plan['cohort_sha256'] or contract['native_size']!=s['native_size']
            or sha(a.references/'references/inputs'/(s['source']['provider_id']+'.rgb'))!=contract['owned_input_sha256']):raise ValueError('cached reference contract differs')
        parsed=row['parsed']['<CAPTION_TO_PHRASE_GROUNDING>'];regions=parsed['bboxes']
        if not regions or len(regions)!=len(parsed['labels']):raise ValueError('preserve unknown reference instead of grading it')
        if any(not(len(b)==4 and all(math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h) for b in regions):raise ValueError('invalid reference geometry')
        regions=[[math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])] for b in regions]
        receipts.append(dict(id=s['id'],boxes=regions,response_sha256=sha(path)))
        boxes=torch.zeros(11,128,4);valid=torch.zeros(11,128,dtype=torch.bool);full=valid.clone();fraction=torch.zeros(11,128);mask=torch.zeros(22,128,dtype=torch.bool);native=[]
        for ri,r in enumerate(RATIOS):
            p,q=map(int,r.split(':'));bank,mw,mh=candidates(w,h,(p,q),[]);native.append(bank)
            boxes[ri,:len(bank)]=torch.tensor(np.array(bank,np.float64)/[w,h,w,h],dtype=torch.float32)
            valid[ri,:len(bank)]=True
            for j,b in enumerate(bank):
                if not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h and (b[2]-b[0])*q==(b[3]-b[1])*p):raise ValueError('invalid exact-ratio native candidate')
                full[ri,j]=all(b[0]<=z[0] and b[1]<=z[1] and b[2]>=z[2] and b[3]>=z[3] for z in regions)
                fraction[ri,j]=min(retained_fraction(b,z) for z in regions)
            for mode in range(2):
                chosen=[j for j,b in enumerate(bank) if (b[2]-b[0]<mw if mode else b[2]-b[0]==mw)]
                if not chosen and mode:chosen=list(range(len(bank)))
                mask[ri*2+mode,chosen]=True
        all_boxes.append(boxes);valids.append(valid);contained.append(full);fractions.append(fraction);eligible.append(mask);native_banks.append(native)
    if sum(len(r['boxes']) for r in receipts)!=23:raise ValueError('all twenty-three previously cached regions required')
    image=torch.stack(images).cuda();bounds=torch.tensor(content).cuda();aspect=torch.tensor(aspects).cuda();boxes=torch.stack(all_boxes).cuda()
    model=LocalCropRanker().cuda().eval();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model'])
    features=model.ranker.encode(image);original=model.ranker.scorer;model.ranker.scorer=nn.Identity()
    vectors,permuted,scores=[],[],[];max_error=0.
    for ri,r in enumerate(RATIOS):
        p,q=map(int,r.split(':'));target=boxes.new_full((15,128),p/q);mode=boxes.new_zeros(15,128);face=boxes.new_zeros(15,6)
        arguments=(boxes[:,ri],aspect,target,mode,face,bounds)
        value=model.ranker.raw_scores(features,*arguments)
        null=model.ranker.raw_scores(tuple(f[donors] for f in features),*arguments)
        if value.shape!=(15,128,762) or not bool(torch.isfinite(value).all() and torch.isfinite(null).all()):raise ValueError('invalid supplementary descriptor layout')
        vectors.append(value);permuted.append(null);score=original(value)[...,0];scores.append(score)
        mode1=boxes.new_ones(15,128)
        scores.append(original(model.ranker.raw_scores(features,boxes[:,ri],aspect,target,mode1,face,bounds))[...,0])
        model.ranker.scorer=original;reference=model.ranker.raw_scores(features,*arguments)[...,0];model.ranker.scorer=nn.Identity()
        max_error=max(max_error,float((score-reference).abs().max()))
    if max_error!=0:raise ValueError('original scorer bits differ under pooled-descriptor replay')
    data=dict(image=torch.stack(vectors,1),permuted=torch.stack(permuted,1),scores=torch.stack(scores,1),
        bank_valid=torch.stack(valids).cuda(),contained=torch.stack(contained).cuda(),fraction=torch.stack(fractions).cuda(),
        eligible=torch.stack(eligible).cuda(),ratios=RATIOS)
    torch.save(dict({k:v.cpu() if torch.is_tensor(v) else v for k,v in data.items()},sources=sources,
        boxes=torch.stack(all_boxes),native_banks=native_banks,reference_receipts=receipts,training_eligible=False),a.root/'inputs.pt')
    results=[];families=defaultdict(lambda:defaultdict(list))
    for binding in plan['fixed_models']:
        path=Path(binding['path'])
        if sha(path)!=binding['sha256']:raise ValueError('head changed after fixed-selection plan')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['arm']!=binding['arm'] or state['seed']!=binding['seed'] or state['epoch']!=binding['selected_epoch']:raise ValueError('selected state metadata differs')
        head=nn.Sequential(nn.Linear(762,96),nn.ReLU(),nn.Linear(96,2)).cuda();head.load_state_dict(state['head']);head.eval()
        metrics,rows=evaluate(head,data,list(range(15)),binding['arm'],sources)
        indices=torch.arange(15,device='cuda')[:,None].expand(-1,11).reshape(-1);queries=torch.arange(11,device='cuda')[None].expand(15,-1).reshape(-1)
        output=head(vector(data,indices,queries,binding['arm'])).reshape(15,11,128,2);valid=data['bank_valid']
        brier=(((output[...,0].sigmoid()-data['contained'].float()).square()*valid).sum(-1)/valid.sum(-1)).mean(-1).cpu().tolist()
        mae=(((output[...,1].sigmoid()-data['fraction']).abs()*valid).sum(-1)/valid.sum(-1)).mean(-1).cpu().tolist()
        for i,s in enumerate(sources):families[binding['arm']][s['family_id']].append(dict(containment_brier=brier[i],fraction_mae=mae[i]))
        results.append(dict(binding,metrics=metrics,predictions=rows,by_source=[dict(id=s['id'],containment_brier=brier[i],fraction_mae=mae[i]) for i,s in enumerate(sources)]))
        progress.write_text(json.dumps(dict(pid=os.getpid(),stage='evaluating',completed_models=len(results),required_models=9,checked_unix=time.time()))+'\n')
    family={arm:{f:{k:statistics.mean(v[k] for v in values) for k in values[0]} for f,values in groups.items()} for arm,groups in families.items()}
    if any(len(v)!=15 for v in family.values()):raise ValueError('fifteen paired development families required')
    means={arm:{k:statistics.mean(r['metrics'][k] for r in results if r['arm']==arm) for k in ['containment_brier','fraction_mae']} for arm in ARMS}
    analysis=dict(models=results,mean_metrics=means,family_metrics=family,paired_image_minus_control={arm:paired(family['image'],family[arm]) for arm in ['geometry','permuted']},
        source_families=15,reference_boxes=23,moved_sources=plan['moved_sources'],original_scorer_max_error=max_error,
        fixed_model_selections_changed=False,new_gradients=False,new_teacher_calls=0,new_test_inference=False,
        face_coverage_certified=False,human_artistic_quality_measured=False,model_promoted=False,paid_model_api_calls=0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    retain(a.root/'analysis.json',analysis)
    if sum(p.stat().st_size for p in a.root.rglob('*') if p.is_file())>256*1024**2:raise ValueError('256MiB supplementary output bound exceeded; preserve files')
    progress.write_text(json.dumps(dict(pid=None,stage='complete',completed_models=9,required_models=9,checked_unix=time.time()))+'\n')
    print(json.dumps({k:v for k,v in analysis.items() if k not in ['models','family_metrics']},indent=2))


if __name__=='__main__':main()
