#!/usr/bin/env python3
"""Render a preselected local-critic witness using only owned RGB and resize."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('preview',type=Path);a=p.parse_args();out=a.root/'critic-inputs';out.mkdir(exist_ok=False);images=[];inputs=json.loads((a.root/'inputs.json').read_text());lookup={s['id']:s for s in inputs['images']}
 def render(id,source,w,h,box,metadata):
  raw=source.read_bytes()
  if len(raw)!=w*h*3:raise ValueError('source RGB dimensions differ')
  x,y,cw,ch=box;dw=max(28,round(448*cw/max(cw,ch)/28)*28);dh=max(28,round(448*ch/max(cw,ch)/28)*28)
  value=subprocess.run([str(a.preview.resolve()),str(w),str(h),str(w*3),f'{x},{y},{cw},{ch},{dw},{dh}'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
  if len(value)!=dw*dh*3:raise ValueError('render length')
  path=out/(id+'.rgb8');path.write_bytes(value);images.append({'id':id,'rgb':str(path),'width':dw,'height':dh,'sha256':sha(value),'source_rgb_sha256':sha(raw),'xywh':box,**metadata})
 for id in ['selfie-family-hike','linen-shirt-model','birthday-cake-kitchen','dog-harness-white']:
  s=lookup[id];render(id,Path(s['rgb']),s['width'],s['height'],[0,0,s['width'],s['height']],{'role':'annotation_only','kind':'original','human_rated':False})
 admitted=json.loads(Path(inputs['admitted_path']).read_text())
 # Fixed before any critic result. One contradictory same-task pair per photographic dev source.
 for id in ['11068','11220','11460','11638','12008','12202']:
  s=next(s for s in admitted['sources'] if s['id']==id);groups={}
  for c in s['candidates']:
   if c['weight']>0:groups.setdefault(c['task_id'],[]).append(c)
  tid=next(t for t,cs in groups.items() if {c['acceptable'] for c in cs}=={True,False})
  for accepted in [True,False]:
   c=next(c for c in groups[tid] if c['acceptable']==accepted);render(tid+'-'+c['candidate_id'],Path(s['tensor_path']).parent/'source.rgb',s['width'],s['height'],c['xywh'],{'role':'dev','kind':'candidate','task_id':tid,'candidate_id':c['candidate_id'],'weak_teacher_acceptable':accepted,'human_rated':False})
 plan={'images':images,'selection':'four AI originals and first contradictory judged candidate pair in manifest order for six fixed photographic TRAIN-development sources; no canonical holdouts','preview_sha256':sha(a.preview.read_bytes()),'pixel_bound':448*448,'new_supervision_admitted':False}
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'images':len(images),'max_pixels':max(r['width']*r['height'] for r in images)}))
if __name__=='__main__':main()
