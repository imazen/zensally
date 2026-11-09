#!/usr/bin/env python3
"""Bounded offline PhotoFramer suitability, with immutable raw response caching."""
import argparse,atexit,hashlib,json,os,re,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration

PROMPT='How would you rate the composition quality of this image? First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. The final answer should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer in JSON format with the following keys: "rating": The score.'

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def protocol_for(model,plan,keep_vision_bf16,precision):
 result={'plan_sha256':sha(plan),'model_assets_sha256':sha(model/'asset-identities.json'),'prompt':PROMPT,'quantization':'NF4 double quantization, BF16 compute; SDPA','vision_bf16':keep_vision_bf16,'loader_adjustment':'old config use_cache null -> true in derived processor config only; model config supplied explicitly','max_new_tokens':512,'max_pixels':448*448,'seed':31,'no_paid_model_api':True,'no_human_quality_claim':True}
 if precision=='bf16':result.update(quantization='none; unquantized BF16 weights and compute; SDPA',precision='bf16',weight_placement='Accelerate auto; GPU12GiB/CPU8GiB; no disk',vision_bf16=True)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('plan',type=Path);p.add_argument('output',type=Path);p.add_argument('--limit',type=int);p.add_argument('--keep-vision-bf16',action='store_true');p.add_argument('--precision',choices=['nf4','bf16'],default='nf4');a=p.parse_args()
 os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8 if a.precision=='bf16' else .65);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads(a.plan.read_text());rows=plan['images'][:a.limit];a.output.mkdir(exist_ok=True)
 protocol=protocol_for(a.model,a.plan,a.keep_vision_bf16,a.precision)
 protocol_path=a.output/'protocol.json'
 if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('cache protocol differs')
 if not protocol_path.exists():protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
 status={'pid':os.getpid(),'stage':'verify-assets','completed':0,'required':len(rows)}
 def progress(**kw):status.update(kw,last_checked_unix=time.time());(a.output/'progress.json').write_text(json.dumps(status))
 def failed():
  if status['pid'] is not None:progress(pid=None,stage='failed-artifacts-retained')
 atexit.register(failed);progress()
 if a.precision=='bf16':
  if protocol['model_assets_sha256']!='aab4ad8df6e3e4bb45e0e69f10e1a0c1ac0c723ff7704ca05e6cd49ace26c570' or protocol['plan_sha256']!='51b788924c962a93ac9e8ec069ec72a147bf08baa06e7fdb5342e2153cc5b901':raise ValueError('registered reference model/plan differs')
  assets=json.loads((a.model/'asset-identities.json').read_text())
  for member in assets['files']:
   path=a.model/member['name']
   if not path.resolve().is_relative_to(a.model.resolve()) or path.stat().st_size!=member['bytes'] or sha(path)!=member['sha256']:raise ValueError('pinned reference model member differs')
 original=json.loads((a.model/'config.json').read_text())
 if original.get('use_cache') is not None:raise ValueError('unexpected source config; reconsider loader adjustment')
 original['use_cache']=True;loader=a.output/'processor';loader.mkdir(exist_ok=True)
 for f in a.model.iterdir():
  if f.suffix in ['.json','.txt'] and f.name not in ['config.json','asset-identities.json','progress.json','model.safetensors.index.json']:
   target=loader/f.name
   if not target.exists():target.symlink_to(f.resolve())
 config_path=loader/'config.json'
 if not config_path.exists():config_path.write_text(json.dumps(original,indent=2)+'\n')
 if json.loads(config_path.read_text())!=original:raise ValueError('derived loader config changed')
 processor=AutoProcessor.from_pretrained(loader,local_files_only=True)
 progress(stage='load-model')
 if a.precision=='nf4':
  quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['model.visual','lm_head'] if a.keep_vision_bf16 else None)
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,config=Qwen2_5_VLConfig.from_dict(original),local_files_only=True,quantization_config=quant,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
 else:
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,config=Qwen2_5_VLConfig.from_dict(original),local_files_only=True,dtype=torch.bfloat16,device_map='auto',max_memory={0:'12GiB','cpu':'8GiB'},attn_implementation='sdpa').eval()
  if any(type(p).__name__=='Params4bit' for p in model.parameters()) or 'disk' in model.hf_device_map.values():raise ValueError('unquantized/no-disk reference contract differs')
  placement={'modules':model.hf_device_map,'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),'quantized':False};target=a.output/'placement.json'
  if target.exists() and json.loads(target.read_text())!=placement:raise ValueError('reference placement changed')
  if not target.exists():target.write_text(json.dumps(placement,indent=2)+'\n')
 if a.keep_vision_bf16:
  if any(type(p).__name__=='Params4bit' for p in model.model.visual.parameters()):raise ValueError('vision encoder was still quantized')
 torch.manual_seed(31);results=[]
 for i,row in enumerate(rows):
  path=a.output/(row['id']+'.json')
  if path.exists():
   prior=json.loads(path.read_text())
   if prior['input']!=row:raise ValueError('cached input differs')
   results.append(prior);continue
  rgb=Path(row['rgb']);w,h=row['width'],row['height']
  if w%28 or h%28 or w*h>448*448 or sha(rgb)!=row['sha256']:raise ValueError('RGB identity or pixel bound')
  raw=np.frombuffer(rgb.read_bytes(),dtype=np.uint8).reshape(h,w,3).copy()
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}];text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=processor(text=[text],images=[raw],do_resize=False,return_tensors='pt').to('cuda')
  grid=inputs['image_grid_thw'].cpu().tolist()
  if grid!=[[1,h//14,w//14]]:raise ValueError('processor changed owned dimensions')
  progress(stage='critic inference',id=row['id'],completed=len(results))
  with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True)
  generated=tokens[:,inputs.input_ids.shape[1]:];answer=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  matches=re.findall(r'<answer>\s*(.*?)\s*</answer>',answer,re.S);rating=None;error=None
  try:
   if len(matches)!=1:raise ValueError('missing or multiple answer tags')
   rating=json.loads(matches[0])['rating']
   if isinstance(rating,bool) or not isinstance(rating,(int,float)) or not 1<=rating<=5:raise ValueError('rating outside range')
  except (ValueError,KeyError,TypeError) as e:error=str(e);rating=None
  result={'input':row,'response':answer,'rating':rating,'parse_error':error,'input_tokens':inputs.input_ids.shape[1],'output_tokens':generated.shape[1],'image_grid_thw':grid,'token_limit_reached':generated.shape[1]==512,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}
  with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  results.append(result);progress(completed=len(results));print(json.dumps({'id':row['id'],'rating':rating,'parse_error':error,'tokens':generated.shape[1]}),flush=True)
 (a.output/'summary.json').write_text(json.dumps({'required':len(rows),'completed':len(results),'parsed':sum(r['rating'] is not None for r in results),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()},indent=2)+'\n');progress(pid=None,completed=len(results),stage='complete')

if __name__=='__main__':main()
