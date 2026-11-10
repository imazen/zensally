#!/usr/bin/env python3
"""Pin existing owned RGB fixtures; do not decode sources or open canonical holdouts."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 base=Path('/mnt/v/output/zensally');images=[]
 for variant in ['pilot','commerce-v2','products-selfies-v3']:
  root=base/'sunburst-jxl'/('2026-11-08-'+variant)
  for r in json.loads((root/'summary.json').read_text())['records']:
   raw=root/r['id']/'source.rgb8'
   if sha(raw)!=r['rgb_sha256']:raise ValueError('AI source changed')
   png=root/r['id']/'lossless-oracle.png'
   images.append({'id':r['id'],'width':r['width'],'height':r['height'],'rgb':str(raw),'rgb_sha256':r['rgb_sha256'],'display':str(png),'synthetic':True,'role':'unadmitted','benchmark':r['id'] in ['selfie-couple-night','selfie-family-hike','dog-harness-white','speaker-pair-white','watch-face-context','selfie-mirror-outfit']})
 admitted=json.loads((base/'crop-fitting/2026-11-08/admitted.json').read_text())
 for s in admitted['sources']:
  if s['use'] not in ['fit','dev']:continue
  root=Path(s['tensor_path']).parent;meta=json.loads((root/'source.json').read_text())
  if meta['category'] not in ['people','general','nature','interiors','food']:continue
  with Image.open(root/'preview.png') as im:w,h=im.size
  raw=root/'preview.rgb'
  if raw.stat().st_size!=w*h*3:raise ValueError('owned preview size differs')
  images.append({'id':'real-'+s['id'],'width':w,'height':h,'rgb':str(raw),'rgb_sha256':sha(raw),'display':str(root/'preview.png'),'synthetic':False,'role':s['use'],'family_id':s['family_id'],'category':meta['category'],'benchmark':False})
 ids=[r['id'] for r in images]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate source ID')
 models=[];modelroot=base/'local-evaluation-models/2026-11-08'
 for r in json.loads((modelroot/'inventory.json').read_text()):
  models.append({'id':Path(r['file']).stem,'path':str(modelroot/r['file']),'sha256':r['sha256'],'bytes':r['bytes']})
 doc={'images':images,'models':models,'purpose':'runtime geometry diagnostics only; no new labels/fits or canonical holdouts','source_parent':'1041c052939284eb7033f21fc44708e864c5f8cc'}
 (a.output/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
 (a.output/'inputs.tsv').write_text(''.join(f"{r['id']}\t{r['width']}\t{r['height']}\t{r['rgb']}\t{int(r['benchmark'])}\n" for r in images))
 print(json.dumps({'images':len(images),'synthetic':sum(r['synthetic'] for r in images),'benchmark':sum(r['benchmark'] for r in images),'models':len(models)}))
if __name__=='__main__':main()
