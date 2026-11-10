#!/usr/bin/env python3
"""Local text-conditioned product boxes on eight fixed witnesses, no paid API."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from fit_heads import sha

QUERIES={
 'linen-shirt-model':'a shirt. a clothing label.',
 'leather-tote-model':'a leather handbag. a handbag strap. a brand label.',
 'dog-harness-white':'a dog. a dog harness. a person.',
 'coffee-bar-context':'a coffee machine. a person.',
 'garden-furniture-context':'a wooden garden table. a garden bench. a dog. a person.',
 'birthday-cake-kitchen':'a birthday cake. a person.',
 'selfie-family-hike':'a backpack. a dog. a person.',
 'vacation-museum-cafe':'an ice cream cone. a paper map. a person.',
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('first_gate',type=Path);p.add_argument('preview',type=Path);p.add_argument('output',type=Path);p.add_argument('--query-plan',type=Path);a=p.parse_args();queries=json.loads(a.query_plan.read_text())['queries'] if a.query_plan else QUERIES;a.output.mkdir(exist_ok=False);os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 processor=AutoProcessor.from_pretrained(a.root/'grounding-dino',local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(a.root/'grounding-dino',local_files_only=True,disable_custom_kernels=True).cuda().eval();lookup={s['id']:s for s in json.loads((a.first_gate/'inputs.json').read_text())['images']};rows=[]
 protocol={'queries':queries,'threshold':.3,'text_threshold':.25,'weights_sha256':sha(a.root/'grounding-dino/model.safetensors'),'asset_identity_sha256':sha(a.root/'grounding-dino/asset-identities.json'),'preview_sha256':sha(a.preview),'preprocess':'owned zenresize Lanczos aspect-preserved long axis1024, do_resize=False; differs from official processor resize','precision':'FP32 TF32 disabled; custom deformable kernels disabled, Torch reference','max_pixels':1024*1024,'canonical_holdouts_opened':False,'new_training_labels_admitted':False}
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 for id,query in queries.items():
  s=lookup[id];w,h=s['width'],s['height'];dw,dh=round(w*1024/max(w,h)),round(h*1024/max(w,h));path=Path(s['rgb'])
  if sha(path)!=s['rgb_sha256']:raise ValueError('owned source RGB differs')
  raw=subprocess.run([str(a.preview.resolve()),str(w),str(h),str(w*3),f'0,0,{w},{h},{dw},{dh}'],input=path.read_bytes(),capture_output=True,check=True).stdout;owned=a.output/(id+'.rgb8');owned.write_bytes(raw);image=np.frombuffer(raw,dtype=np.uint8).reshape(dh,dw,3).copy();inputs=processor(images=image,text=query,do_resize=False,return_tensors='pt').to('cuda')
  if list(inputs.pixel_values.shape[-2:])!=[dh,dw]:raise ValueError('processor changed owned dimensions')
  with torch.inference_mode():outputs=model(**inputs)
  result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.3,text_threshold=.25,target_sizes=[(h,w)])[0];detections=[]
  for box,score,label in zip(result['boxes'],result['scores'],result['text_labels']):
   pixel=box.cpu().tolist();normalized=(box/box.new_tensor([w,h,w,h])).cpu().tolist()
   if not np.isfinite(normalized).all() or not np.isfinite(float(score)):raise ValueError('invalid localizer output')
   visible=np.clip(normalized,0,1).tolist();detections.append({'box_source_pixels':pixel,'box_normalized_unclamped':normalized,'visible_box_normalized':visible,'score':float(score),'label':label})
  row={'id':id,'source_rgb_sha256':s['rgb_sha256'],'owned_input_sha256':sha(owned),'source_size':[w,h],'input_size':[dw,dh],'query':query,'detections':detections,'face_coverage_certified':False,'training_eligible':False};rows.append(row)
  with (a.output/(id+'.json')).open('x') as f:json.dump(row,f,indent=2);f.write('\n')
  (a.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'completed':len(rows),'required':len(queries),'checked_unix':time.time()}));print(json.dumps({'id':id,'detections':len(detections),'labels':[d['label'] for d in detections]}),flush=True)
 (a.output/'summary.json').write_text(json.dumps({'completed':len(rows),'required':len(queries),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved(),'parameters':sum(p.numel() for p in model.parameters())},indent=2)+'\n');(a.output/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed':len(rows),'required':len(queries),'checked_unix':time.time()}))
if __name__=='__main__':main()
