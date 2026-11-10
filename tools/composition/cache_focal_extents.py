#!/usr/bin/env python3
"""Stronger frozen visual descriptors for the same primary-extent targets."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from fit_heads import sha

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('root',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();out=a.output/'extent-cache';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(a.models/'focalnet-source/src'));from focalnet.ranking import load_human_checkpoint
 cache_path=a.root/'extent-cache/features.pt';cache=torch.load(cache_path,map_location='cpu',weights_only=False);aux_path=a.auxiliary/'auxiliary.pt'
 if sha(aux_path)!=cache['auxiliary_source_cache_sha256']:raise ValueError('source cache changed')
 aux=torch.load(aux_path,map_location='cpu',weights_only=False);model_path=a.models/'focalnet/focalnet-human.pt';model,_=load_human_checkpoint(model_path);model.cuda().eval();means=torch.tensor([.485,.456,.406],device='cuda')[:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None];vectors=[]
 for start in range(0,56,16):
  original=aux['image'][start:start+16].cuda();normalized=(original-means)/std;image=torch.zeros_like(normalized)
  for j,valid in enumerate(aux['packed']['valid_rect'][start:start+16]):
   x0,y0,x1,y1=(valid*256).round().int().tolist();image[j,:,y0:y1,x0:x1]=normalized[j,:,y0:y1,x0:x1]
  fused=model.base.feature_pyramid(image);rank=model.rank_projection(fused);vector=torch.cat((F.adaptive_avg_pool2d(rank,(4,4)).flatten(1),F.adaptive_avg_pool2d(fused,(2,2)).flatten(1),fused.mean((-2,-1)),torch.zeros(len(image),16,device='cuda')),-1)
  if vector.shape[1]!=512 or not bool(torch.isfinite(vector).all()):raise ValueError('invalid frozen descriptor')
  vectors.append(vector.cpu())
 cache['visual']=torch.cat(vectors);cache['parent_extent_cache_sha256']=sha(cache_path);cache['encoder_sha256']=sha(model_path);cache['descriptor_origin']='FocalNet rank16 pool4x4 + fused48 pool2x2 + global48 + zeros16, owned Lanczos/ImageNet content and zero normalized padding';torch.save(cache,out/'features.pt');metadata={k:v for k,v in cache.items() if k not in ['visual','geometry','targets','present']};metadata.update(cache_sha256=sha(out/'features.pt'),encoder_parameters=sum(p.numel() for p in model.parameters()),canonical_holdouts_opened=False,diagnostic_maps_saved=False);(out/'admission.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'sources':len(cache['sources']),'descriptor_shape':list(cache['visual'].shape),'cache_sha256':metadata['cache_sha256'],'encoder_parameters':metadata['encoder_parameters']}))
if __name__=='__main__':main()
