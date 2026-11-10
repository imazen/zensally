#!/usr/bin/env python3
"""Admit only synthetic geometry targets under the registered family protocol."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import numpy as np
import torch
from composition_features import candidate_features,NAMES

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('preview',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);jobs=[];plan_identities={}
 for name in ['sunburst-ecommerce','sunburst-commerce-v2','sunburst-products-selfies-v3']:
  path=Path('/mnt/v/output/zensally')/name/'2026-11-08/plan.json';jobs.extend(json.loads(path.read_text()));plan_identities[str(path)]=sha(path.read_bytes())
 jobs={j['id']:j for j in jobs};sources=[];tensors=[];packed={k:[] for k in ['boxes','source_aspect','target_aspect','mode','face_context','valid_rect','targets','mask']}
 for s in json.loads((a.root/'inputs.json').read_text())['images']:
  if not s['synthetic']:continue
  j=jobs[s['id']];family='brand-'+j['brand_requested'].lower() if j.get('brand_requested') else 'scenario-'+j['scenario_family'];use='dev' if int(sha(('geometry-distill-v1:'+family).encode())[:8],16)%4==0 else 'fit';w,h=s['width'],s['height'];raw=Path(s['rgb']).read_bytes()
  if len(raw)!=w*h*3 or sha(raw)!=s['rgb_sha256']:raise ValueError('source RGB differs')
  tw,th=max(1,round(w*256/max(w,h))),max(1,round(h*256/max(w,h)));resized=subprocess.run([str(a.preview.resolve()),str(w),str(h),str(w*3),f'0,0,{w},{h},{tw},{th}'],input=raw,capture_output=True,check=True).stdout
  if len(resized)!=tw*th*3:raise ValueError('resize length')
  image=np.zeros((256,256,3),dtype=np.uint8);ox,oy=(256-tw)//2,(256-th)//2;image[oy:oy+th,ox:ox+tw]=np.frombuffer(resized,dtype=np.uint8).reshape(th,tw,3);tensors.append(torch.from_numpy(image).permute(2,0,1).float()/255)
  boxes=[];aspects=[];modes=[]
  for num,den in [(1,1),(4,3),(3,4),(16,9),(9,16),(2,1),(1,2),(3,1),(1,3),(6,1),(1,6)]:
   aspect=num/den;cw=min(w,h*aspect);ch=cw/aspect
   for mode,amount in [(0,1.),(1,.75)]:
    bw,bh=cw*amount,ch*amount
    for t in [0,.25,.5,.75,1]:
     x,y=(w-bw)*t,(h-bh)*t;boxes.append([x/w,y/h,(x+bw)/w,(y+bh)/h]);aspects.append(aspect);modes.append(mode)
  boxes.extend([[0,0,1,1]]*(128-len(boxes)));aspects.extend([1.]*(128-len(aspects)));modes.extend([0.]*(128-len(modes)));teachers=[];identities={}
  for part,suffix in [('geometry','-pose'),('gaze-all',''),('scan-all','')]:
   path=a.root/part/(s['id']+suffix+'.json');value=json.loads(path.read_text())
   if value['rgb_sha256']!=s['rgb_sha256']:raise ValueError('teacher RGB mismatch')
   teachers.append(value['result']);identities[str(path)]=sha(path.read_bytes())
  targets=candidate_features(np.asarray(boxes),*teachers);mask=np.ones((128,32),np.float32);mask[110:]=0
  if not teachers[0]['landmarks']:mask[:,:16]=0
  if not teachers[1]['people']:mask[:,16:24]=0
  values={'boxes':boxes,'source_aspect':w/h,'target_aspect':aspects,'mode':modes,'face_context':[0,0,0,0,0,0],'valid_rect':[ox/256,oy/256,(ox+tw)/256,(oy+th)/256],'targets':targets,'mask':mask}
  for key,value in values.items():packed[key].append(torch.as_tensor(value,dtype=torch.float32))
  sources.append({'id':s['id'],'family_id':family,'use':use,'rgb_sha256':s['rgb_sha256'],'image_tensor_sha256':sha(image.tobytes()),'teacher_identities':identities,'synthetic':True,'human_rated':False,'face_safety_certified':False,'composition_training_eligible':False,'geometry_training_eligible':use=='fit'})
 if (sum(s['use']=='fit' for s in sources),sum(s['use']=='dev' for s in sources))!=(48,8):raise ValueError('registered roles differ')
 if {s['family_id'] for s in sources if s['use']=='fit'}&{s['family_id'] for s in sources if s['use']=='dev'}:raise ValueError('family role overlap')
 cache={'sources':sources,'image':torch.stack(tensors),'packed':{k:torch.stack(v) for k,v in packed.items()},'feature_names':NAMES,'plan_identities':plan_identities,'preview_sha256':sha(a.preview.read_bytes()),'canonical_holdouts_opened':False};torch.save(cache,a.output/'auxiliary.pt')
 manifest={k:v for k,v in cache.items() if k not in ['image','packed']};manifest['cache_sha256']=sha((a.output/'auxiliary.pt').read_bytes());manifest['payload_bytes']=sum(t.numel()*t.element_size() for t in [cache['image'],*cache['packed'].values()]);(a.output/'admission.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'fit':48,'dev':8,'payload_bytes':manifest['payload_bytes'],'cache_sha256':manifest['cache_sha256']}))
if __name__=='__main__':main()
