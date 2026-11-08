#!/usr/bin/env python3
"""Warm local graph timing for zenbench; preloaded tensors, reusable binding option."""
import argparse,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

IDS=['dog-harness-white','speaker-pair-white','watch-face-context','selfie-couple-night','selfie-family-hike','selfie-mirror-outfit']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('cache',type=Path);p.add_argument('--backend',choices=['disabled','optimized','bound'],default='disabled');p.add_argument('--repeats',type=int,default=30);p.add_argument('--profile',action='store_true');a=p.parse_args()
 if not 1<=a.repeats<=10000:raise ValueError('bounded repeat count')
 data=torch.load(a.cache,map_location='cpu',weights_only=False);indices={s['id']:i for i,s in enumerate(data['sources'])};images=[];mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None]
 for id in IDS:
  i=indices[id];valid=data['packed']['valid_rect'][i].numpy();x0,y0,x1,y1=np.rint(valid*256).astype(int);image=np.zeros((1,3,256,256),np.float32);image[0,:,y0:y1,x0:x1]=((data['image'][i].numpy()-mean)/std)[:,y0:y1,x0:x1];images.append({'image':image,'boxes':data['packed']['boxes'][i:i+1].numpy(),'content':valid[None]})
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL if a.backend=='disabled' else ort.GraphOptimizationLevel.ORT_ENABLE_ALL;session=ort.InferenceSession(str(a.model),sess_options=options,providers=['CPUExecutionProvider']);outputs=[np.empty((1,1,64,64),np.float32),np.empty((1,128),np.float32)];bindings=[]
 if a.backend=='bound':
  for image in images:
   binding=session.io_binding()
   for name,value in image.items():binding.bind_cpu_input(name,value)
   for name,value in zip(['importance','crop_scores'],outputs):binding.bind_output(name,'cpu',0,np.float32,value.shape,value.ctypes.data)
   bindings.append(binding)
  for image,binding in zip(images,bindings):
   expected=session.run(['importance','crop_scores'],image);session.run_with_iobinding(binding)
   if not all(np.array_equal(x,y) for x,y in zip(outputs,expected)):raise ValueError('context buffers change output bits')
 def run(index):
  if a.backend=='bound':session.run_with_iobinding(bindings[index])
  else:session.run(['importance','crop_scores'],images[index])
 for i in range(6):run(i)
 start=time.perf_counter_ns() if not a.profile else None
 for _ in range(a.repeats):
  for i in range(6):run(i)
 result={'backend':a.backend,'calls':6*a.repeats,'images':6,'threads':1,'scope':'warm CPU ORT forward on preloaded normalized 256-square tensor and 128 candidate slots; both importance and crop scores returned; excludes resize, candidate generation, ranking/face guards, load and IO','context_buffer_bit_equality_checked':a.backend=='bound'}
 if not a.profile:result['inference_ns']=(time.perf_counter_ns()-start)/(6*a.repeats)
 print(('PROFILE ' if a.profile else 'RESULT ')+json.dumps(result))
if __name__=='__main__':main()
