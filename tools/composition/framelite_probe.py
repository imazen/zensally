#!/usr/bin/env python3
"""Pinned local direct-crop teacher on owned RGB; retain failures and raw answers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from fit_heads import sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('models',type=Path);p.add_argument('first_gate',type=Path);p.add_argument('preview',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.models/'framelite-source'));from framelite import FrameLite
 from framelite.model import _generation_prompt
 from framelite.geometry import parse_crop,InvalidCropError
 os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.manual_seed(31)
 model=FrameLite.from_pretrained(a.models/'framelite',device='cuda:0');lookup={s['id']:s for s in json.loads((a.first_gate/'inputs.json').read_text())['images']};ids=json.loads((a.first_gate/'witness/witness.json').read_text())['bounded_witness_sources'];rows=[]
 protocol={'model_assets_sha256':sha(a.models/'asset-identities.json'),'preview_sha256':sha(a.preview),'prompt':model.config['prompt'],'generation':model.config['generation'],'preprocess':'owned zenresize Lanczos, aspect-preserved rounded multiple32, long axis768, do_resize=False; differs from upstream image resize/dependency versions','pixel_bound':768*768,'sources':ids,'canonical_holdouts_opened':False,'new_labels_admitted':False};(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 for id in ids:
  s=lookup[id];w,h=s['width'],s['height'];dw=max(32,round(w*768/max(w,h)/32)*32);dh=max(32,round(h*768/max(w,h)/32)*32);path=Path(s['rgb'])
  if sha(path)!=s['rgb_sha256'] or not 65536<=dw*dh<=768*768:raise ValueError('source identity or model pixel bound')
  raw=subprocess.run([str(a.preview.resolve()),str(w),str(h),str(w*3),f'0,0,{w},{h},{dw},{dh}'],input=path.read_bytes(),capture_output=True,check=True).stdout;owned=a.output/(id+'.rgb8');owned.write_bytes(raw);image=np.frombuffer(raw,dtype=np.uint8).reshape(dh,dw,3).copy();text=_generation_prompt(model.processor,model.config['prompt']);encoded=model.processor(text=[text],images=[image],do_resize=False,padding=True,truncation=False,return_tensors='pt').to('cuda')
  if encoded['image_grid_thw'].cpu().tolist()!=[[1,dh//16,dw//16]]:raise ValueError('processor changed owned dimensions')
  width=encoded.input_ids.shape[1]
  with torch.inference_mode():tokens=model.model.generate(**encoded,**model.config['generation'],eos_token_id=model.processor.tokenizer.eos_token_id,pad_token_id=model.processor.tokenizer.pad_token_id)
  completion=tokens[0,width:];text=model.processor.batch_decode([completion],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0];box=None;error=None
  try:box=list(parse_crop(text))
  except InvalidCropError as e:error=str(e)
  row={'id':id,'rgb_sha256':s['rgb_sha256'],'owned_input_sha256':sha(owned),'raw_text':text,'completion_token_ids':completion.cpu().tolist(),'bbox_normalized':box,'valid':box is not None,'error':error,'input_tokens':width,'output_tokens':len(completion),'source_size':[w,h],'input_size':[dw,dh],'eos_generated':bool((completion==model.processor.tokenizer.eos_token_id).any()),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}
  with (a.output/(id+'.json')).open('x') as f:json.dump(row,f,indent=2);f.write('\n')
  rows.append(row);(a.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'completed':len(rows),'required':len(ids),'checked_unix':time.time()}));print(json.dumps({'id':id,'box':box,'error':error}),flush=True)
 (a.output/'summary.json').write_text(json.dumps({'completed':len(rows),'valid':sum(r['valid'] for r in rows),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()},indent=2)+'\n');(a.output/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed':len(rows),'required':len(ids),'checked_unix':time.time()}))
if __name__=='__main__':main()
