#!/usr/bin/env python3
"""Compact local gaze/scanpath inference; no persisted diagnostic maps."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from teacher_models import load

def image(row,kind,device):
 r=row[kind];b=Path(r['path']).read_bytes()
 if hashlib.sha256(b).hexdigest()!=r['sha256'] or len(b)!=r['width']*r['height']*3:raise ValueError('frozen model pixels changed')
 value=torch.from_numpy(np.frombuffer(b,dtype=np.uint8).copy().reshape(r['height'],r['width'],3)).permute(2,0,1)[None].float().to(device)
 return value

def heads(row,geometry):
 entries=[]
 for d in geometry['result']['detections']:
  x,y,w,h=d['xywh'];W,H=row['width'],row['height'];box=[max(0,(x-.15*w)/W),max(0,(y-.25*h)/H),min(1,(x+1.15*w)/W),min(1,(y+1.25*h)/H)]
  if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('fully outside face detection; explicit review needed')
  entries.append({'head':box,'face_score':d['categories'][0]['score']})
 entries.sort(key=lambda r:-r['face_score'])
 if len(entries)>16:raise ValueError('person count exceeds registered gaze bound')
 return entries

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['gaze','scan']);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--ids',nargs='+');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(2);torch.manual_seed(31)
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 doc=json.loads((a.root/'inputs.json').read_text());rows=doc['images']
 if a.ids:
  if not set(a.ids)<=set(r['id'] for r in rows):raise ValueError('unknown source ID')
  rows=[r for r in rows if r['id'] in a.ids]
 model=load(a.kind,a.root/'teachers').to(a.device);prior=np.load(a.root/'teachers/centerbias_mit1003.npy') if a.kind=='scan' else None;results=[]
 for row in rows:
  value=image(row,a.kind,a.device);_,_,h,w=value.shape
  if a.kind=='gaze':
   geometry=json.loads((a.root/'geometry'/(row['id']+'-faces.json')).read_text());people=heads(row,geometry);targets=[]
   if people:
    value=value/255;mean=value.new_tensor([.485,.456,.406])[None,:,None,None];std=value.new_tensor([.229,.224,.225])[None,:,None,None];prediction=model({'images':(value-mean)/std,'bboxes':[[p['head'] for p in people]]})
    maps=prediction['heatmap'][0];inout=prediction['inout'][0]
    if not bool(torch.isfinite(maps).all() and torch.isfinite(inout).all()):raise ValueError('nonfinite gaze prediction')
    for i,ppl in enumerate(people):
     m=maps[i];idx=int(m.argmax());targets.append({**ppl,'target':[(idx%m.shape[-1]+.5)/m.shape[-1],(idx//m.shape[-1]+.5)/m.shape[-2]],'target_score':float(m.max()),'in_frame_score':float(inout[i])})
   result={'people':targets,'face_coverage_certified':False}
  else:
   cb=F.interpolate(torch.from_numpy(prior.astype(np.float32))[None,None].to(a.device),size=(h,w),mode='nearest')[0,0];cb=cb-torch.logsumexp(cb.flatten(),0);paths=[[(w/2,h/2)] for _ in range(4)];logs=[[] for _ in paths];generator=torch.Generator(device=a.device).manual_seed(31^int(row['rgb_sha256'][:8],16))
   for step in range(6):
    hx=torch.full((4,4),float('nan'),device=a.device);hy=hx.clone()
    for i,path in enumerate(paths):
     for age,(x,y) in enumerate(path[::-1][:4]):hx[i,age]=x;hy[i,age]=y
    density=model(value.expand(4,-1,-1,-1),cb[None].expand(4,-1,-1),hx,hy)
    if density.shape!=(4,1,h,w) or not bool(torch.isfinite(density).all()):raise ValueError('invalid scanpath density')
    for i in range(4):
     flat=density[i,0].flatten();idx=int(torch.multinomial(flat.exp(),1,generator=generator));paths[i].append((idx%w+.5,idx//w+.5));logs[i].append(float(flat[idx]))
   result={'initial_fixation':'image center, imposed before first free fixation','paths':[[{'x':x/w,'y':y/h,'log_density':l} for (x,y),l in zip(path[1:],log)] for path,log in zip(paths,logs)],'centerbias':'retained MIT1003 prior resized nearest and renormalized','diagnostic_maps_saved':False}
  payload={'id':row['id'],'role':row['role'],'family_id':row['family_id'],'rgb_sha256':row['rgb_sha256'],'kind':a.kind,'result':result}
  with (a.output/(row['id']+'.json')).open('x') as f:json.dump(payload,f);f.write('\n')
  results.append(row['id']);(a.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':a.kind,'completed':len(results),'required':len(rows),'checked_unix':time.time()}))
  print(json.dumps({'kind':a.kind,'completed':len(results),'required':len(rows),'id':row['id']}),flush=True)
 if a.device=='cuda':torch.cuda.synchronize()
 summary={'completed':len(results),'required':len(rows),'kind':a.kind,'torch':torch.__version__,'cuda_peak_allocated':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda' else None,'diagnostic_maps_saved':False}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
