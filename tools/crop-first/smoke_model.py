#!/usr/bin/env python3
"""Record an untrained full-input shape/constraint probe, not a quality score."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from crop_ranker import CropRanker
from crop_policy import Rect,plan_crops
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
torch.manual_seed(19);torch.set_num_threads(1)
m=CropRanker().eval();w,h=1280,720;face=Rect(880,180,1000,340)
boxes=[];ratios=[];modes=[];task_ids=[]
for mode in ['minimal','maximal']:
 for rw,rh in [(1,1),(16,9),(9,16),(4,1),(1,4)]:
  plan=plan_crops(w,h,(rw,rh),mode=mode,protected_faces=(face,),faces_reviewed=True)
  for r in plan.candidates:
   assert r.contains(face)
   boxes.append([r.left/w,r.top/h,r.right/w,r.bottom/h]);ratios.append(rw/rh);modes.append(float(mode=='maximal'));task_ids.append(f'{rw}:{rh}/{mode}')
with torch.no_grad():
 encoded=m.encode(torch.rand(1,3,256,256))
 scores=m.score(encoded,torch.tensor([boxes]),torch.tensor([w/h]),torch.tensor([ratios]),torch.tensor([modes]),torch.tensor([[1.,face.left/w,face.top/h,face.right/w,face.bottom/h,1.]]),torch.ones(1,len(boxes),dtype=torch.bool),torch.tensor([[0.,56/256,1.,200/256]]))
assert torch.isfinite(scores).all()
result={'role':'untrained architecture smoke only','torch':torch.__version__,'parameters':sum(p.numel() for p in m.parameters()),'encoder_parameters':sum(p.numel() for name,p in m.named_parameters() if not name.startswith('scorer.')),'scorer_parameters':sum(p.numel() for p in m.scorer.parameters()),'input_shape':[1,3,256,256],'encoded_shape':list(encoded.shape),'candidate_count':len(boxes),'task_count':len(set(task_ids)),'scores_shape':list(scores.shape),'all_scores_finite':True,'all_candidates_contain_synthetic_face':True,'trained':False,'onnx_exported':False,'rust_runtime_implemented':False,'quality_measured':False,'latency_measured':False}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
