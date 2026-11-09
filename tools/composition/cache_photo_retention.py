#!/usr/bin/env python3
"""Cache source-conditioned crop features without exposing offline region inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn

from artifact_identity import sha
from fit_extents import retain
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['rank_cache','regions','weights','old_protocol','predictions','output']:
        p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.rank_cache,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),
        (a.regions,'101110157703882de1020b24df012d33f7a9adfd7b02cfb5c2f5588366c7918d'),
        (a.weights,'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df')]:
        if sha(path)!=digest:raise ValueError('registered photographic input/known-box/encoder differs')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rank=torch.load(a.rank_cache,map_location='cpu',weights_only=False)
    regions=torch.load(a.regions,map_location='cpu',weights_only=False)
    if [s['id'] for s in rank['sources']]!=[s['id'] for s in regions['sources']]:
        raise ValueError('source ordering differs')
    indices=regions['known'].nonzero().flatten().tolist()
    sources=[rank['sources'][i] for i in indices]
    if len(indices)!=87 or sum(s['role']=='fit' for s in sources)!=72 or sum(s['role']=='dev' for s in sources)!=15:
        raise ValueError('registered72/15 known-box sources required')
    if {s['family_id'] for s in sources if s['role']=='fit'} & {s['family_id'] for s in sources if s['role']=='dev'}:
        raise ValueError('fitting and development families overlap')
    protocol=json.loads(a.old_protocol.read_text())
    donors=protocol['donors']
    if len(donors)!=119:raise ValueError('original fixed shuffled-input mapping required')
    for i,j in enumerate(donors):
        if (rank['sources'][i]['role']!=rank['sources'][j]['role'] or
            not torch.equal(rank['content'][i],rank['content'][j])):
            raise ValueError('donor crosses role or exact letterbox shape')
    a.output.mkdir(parents=True,exist_ok=False)
    model=LocalCropRanker().cuda().eval()
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['model'])
    image=rank['image'].cuda()
    chunks=[model.ranker.encode(image[i:i+16]) for i in range(0,119,16)]
    features=tuple(torch.cat([chunk[level] for chunk in chunks]) for level in range(2))
    original=model.ranker.scorer
    initial_head={k:v.cpu().clone() for k,v in original[0].state_dict().items()}
    model.ranker.scorer=nn.Identity()
    content=rank['content'].cuda()
    aspect=torch.tensor([s['native_size'][0]/s['native_size'][1] for s in rank['sources']],device='cuda')
    boxes=rank['boxes'].cuda()
    actual,shuffled,rank_scores=[],[],[]
    max_score_error=0.
    for start in range(0,87,16):
        batch=indices[start:start+16]
        source_features=tuple(f[batch] for f in features)
        donor_features=tuple(f[[donors[i] for i in batch]] for f in features)
        visual,permuted,scores=[],[],[]
        for ri,ratio in enumerate(rank['ratios']):
            count=boxes.shape[2]
            target=boxes.new_full((len(batch),count),int(ratio.split(':')[0])/int(ratio.split(':')[1]))
            mode=boxes.new_zeros(len(batch),count)
            arguments=(boxes[batch,ri],aspect[batch],target,mode,boxes.new_zeros(len(batch),6),content[batch])
            vector=model.ranker.raw_scores(source_features,*arguments)
            donor=model.ranker.raw_scores(donor_features,*arguments)
            if vector.shape[-1]!=762 or not bool(torch.isfinite(vector).all() and torch.isfinite(donor).all()):
                raise ValueError('finite registered pooled descriptor layout required')
            visual.append(vector.cpu());permuted.append(donor.cpu())
            score0=original(vector)[...,0]
            mode1=boxes.new_ones(len(batch),count)
            vector1=model.ranker.raw_scores(source_features,boxes[batch,ri],aspect[batch],target,mode1,
                boxes.new_zeros(len(batch),6),content[batch])
            scores.extend([score0.cpu(),original(vector1)[...,0].cpu()])
            model.ranker.scorer=original
            reference=model.ranker.raw_scores(source_features,*arguments)[...,0]
            model.ranker.scorer=nn.Identity()
            error=float((score0-reference).abs().max())
            max_score_error=max(error,max_score_error)
            if error!=0:raise ValueError('pooled descriptor replay must retain original score bits')
        actual.append(torch.stack(visual,1));shuffled.append(torch.stack(permuted,1));rank_scores.append(torch.stack(scores,1))
        (a.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),stage='pooling',completed=min(start+16,87),required=87,checked_unix=time.time()))+'\n')
    score=torch.cat(rank_scores)
    lookup={s['id']:i for i,s in enumerate(sources)}
    motion=[]
    for row in json.loads(a.predictions.read_text()):
        if row['id'] not in lookup or not row['query_valid']:continue
        j=lookup[row['id']];i=indices[j];q=row['query'];eligible=rank['eligible'][i,q]
        selected=int(score[j,q].masked_fill(~eligible,-torch.inf).argmax())
        edge=float((rank['boxes'][i,q//2,selected]-rank['boxes'][i,q//2,row['selected']]).abs().max())*128
        if edge>3:raise ValueError('original development crop replay exceeds unchanged edge budget')
        motion.append(edge)
    if len(motion)!=329:raise ValueError('all known informative original development decisions required')
    payload=dict(image=torch.cat(actual),permuted=torch.cat(shuffled),scores=score,sources=sources,
        eligible=rank['eligible'][indices],bank_valid=regions['bank_valid'][indices],
        contained=regions['contained'][indices],fraction=regions['fraction'][indices],
        boxes=rank['boxes'][indices],ratios=rank['ratios'],initial_head=initial_head,visual_dimensions=704,
        original_donors=donors,rank_cache_sha256=sha(a.rank_cache),region_cache_sha256=sha(a.regions),
        encoder_sha256=sha(a.weights),fit_sources=72,development_sources=15,
        original_donor_protocol_sha256=sha(a.old_protocol),original_predictions_sha256=sha(a.predictions),
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        admission='valid fitting-source noisy known-box containment/fraction only; not complete subjects or faces',
        training_eligible_roles=['fit'],face_coverage_certified=False)
    target=a.output/'features.pt';torch.save(payload,target)
    if target.stat().st_size>1024**3:raise ValueError('one GiB feature-cache bound exceeded; preserve output')
    retain(a.output/'qualification.json',dict(max_original_score_error=max_score_error,
        original_development_decisions=len(motion),max_original_crop_map_edge_difference=max(motion),
        finite_descriptors=True,actual_crop_geometry_preserved_for_donors=True,
        feature_bytes=target.stat().st_size,feature_sha256=sha(target),
        fit_families=len({s['family_id'] for s in sources if s['role']=='fit'}),
        development_families=len({s['family_id'] for s in sources if s['role']=='dev'}),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
    (a.output/'progress.json').write_text(json.dumps(dict(pid=None,stage='complete',completed=87,required=87,checked_unix=time.time()))+'\n')
    print((a.output/'qualification.json').read_text())


if __name__=='__main__':main()
