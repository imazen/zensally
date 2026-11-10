#!/usr/bin/env python3
"""Freeze existing fit/dev inputs and diagnostic AI sources; owned resize only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from PIL import Image

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--preview',type=Path,required=True);a=p.parse_args();base=Path('/mnt/v/output/zensally');admitted=base/'crop-fitting/2026-11-08/admitted.json';old=json.loads(admitted.read_text());images=[]
 for s in old['sources']:
  if s['use'] not in ['fit','dev']:continue
  directory=Path(s['tensor_path']).parent
  with Image.open(directory/'preview.png') as im:w,h=im.size
  images.append({'id':s['id'],'rgb':str(directory/'preview.rgb'),'width':w,'height':h,'role':s['use'],'family_id':s['family_id'],'synthetic':False})
 for s in json.loads((base/'runtime-models/2026-11-08/manifest.json').read_text())['images']:
  if s['synthetic']:images.append({**s,'role':'annotation_only','family_id':'ai-'+s['id']})
 out=a.root/'inputs';out.mkdir(exist_ok=False)
 for s in images:
  raw=Path(s['rgb']).read_bytes();w,h=s['width'],s['height']
  if len(raw)!=w*h*3:raise ValueError('invalid owned RGB')
  s['rgb_sha256']=sha(raw);scale=max(128/min(w,h),512/max(w,h));dw,dh=round(w*scale),round(h*scale)
  if dw*dh>1024*1024:raise ValueError('scanpath input exceeds registered one-megapixel limit')
  config=f'0,0,{w},{h},448,448;0,0,{w},{h},{dw},{dh}'
  resized=subprocess.run([str(a.preview.resolve()),str(w),str(h),str(w*3),config],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
  offset=448*448*3
  if len(resized)!=offset+dw*dh*3:raise ValueError('owned resize output length')
  for name,b,rw,rh in [('gaze',resized[:offset],448,448),('scan',resized[offset:],dw,dh)]:
   path=out/(s['id']+'-'+name+'.rgb8');path.write_bytes(b);s[name]={'path':str(path),'width':rw,'height':rh,'sha256':sha(b)}
 manifest={'schema':1,'admitted_path':str(admitted),'admitted_sha256':sha(admitted.read_bytes()),'preview_binary':str(a.preview.resolve()),'preview_sha256':sha(a.preview.read_bytes()),'preprocess':'owned zenresize Lanczos RGB8; gaze 448-square plus ImageNet normalization; scan aspect-preserved min axis128/max axis>=512 RGB0..255. Distinct from upstream torchvision-resize pipeline; no reference parity claim.','images':images,'canonical_holdouts_opened':False,'diagnostic_maps':False}
 with (a.root/'inputs.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 print(json.dumps({'fit':sum(x['role']=='fit' for x in images),'dev':sum(x['role']=='dev' for x in images),'annotation_only':sum(x['role']=='annotation_only' for x in images),'max_scan_pixels':max(x['scan']['width']*x['scan']['height'] for x in images)}))
if __name__=='__main__':main()
