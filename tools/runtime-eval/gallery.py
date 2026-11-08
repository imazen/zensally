#!/usr/bin/env python3
"""Render evaluation-only thumbnails and saved geometry for visual inspection."""
import argparse,html,json
from pathlib import Path
from PIL import Image,ImageDraw

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root;doc=json.loads((root/'manifest.json').read_text());gallery=root/'gallery';gallery.mkdir(exist_ok=False);assets=gallery/'assets';assets.mkdir();models={}
 for group in ['reference-eval','reference-masks-v2','rust-eval']:
  folder=root/group
  if not folder.exists():continue
  for model in folder.iterdir():
   path=model/'results.json'
   if not path.exists():continue
   result=json.loads(path.read_text());rows=result if isinstance(result,list) else result['rows']
   models[model.name]={r['id']:r['result'] for r in rows}
 pages=['<!doctype html><meta charset="utf-8"><title>Local model evaluation</title><style>body{font:14px system-ui;background:#202124;color:#eee}article{border-top:1px solid #aaa;padding:12px} .models{display:flex;gap:8px;overflow-x:auto}figure{margin:2px;min-width:220px;width:220px}img{max-width:220px;max-height:280px}pre{white-space:pre-wrap}svg{max-width:220px;max-height:280px}</style><h1>76-source local geometry evaluation</h1><p>Diagnostic outputs, not human ground truth. Native aspect ratios preserved. Face models use their recorded thresholds; masks use their own classes. No canonical VALIDATE/TEST opened.</p>']
 contact=[]
 for r in doc['images']:
  source=Image.frombytes('RGB',(r['width'],r['height']),Path(r['rgb']).read_bytes());source.thumbnail((320,320));source.save(assets/(r['id']+'.png'))
  if r['id'].startswith(('selfie-','birthday-','vacation-')) or r['id'] in ['leather-tote-model','linen-shirt-model','jewelry-profile-tall','real-12000','real-12018']:
   card=Image.new('RGB',(320,350),'white');card.paste(source,((320-source.width)//2,20));ImageDraw.Draw(card).text((4,333),r['id'],fill='black');contact.append(card)
  pages.append('<article id="'+html.escape(r['id'])+'"><h2>'+html.escape(r['id'])+'</h2><div class="models">')
  pages.append('<figure>Source<br><img src="assets/'+r['id']+'.png"></figure>')
  for model,results in models.items():
   if r['id'] not in results:continue
   v=results[r['id']];view=Image.open(assets/(r['id']+'.png')).convert('RGB');draw=ImageDraw.Draw(view);label=[]
   boxes=[]
   for f in v.get('faces',[]):
    x1,y1,x2,y2=f['box_percent'];boxes.append([x1*r['width']/100,y1*r['height']/100,(x2-x1)*r['width']/100,(y2-y1)*r['height']/100,str(round(f['confidence'],2))])
   for d in v.get('detections',[]):
    c=d['categories'][0];boxes.append(d['xywh']+[str(c['name'])+' '+str(round(c['score'],2))])
   for x,y,w,h,text in boxes:
    sx,sy=view.width/r['width'],view.height/r['height'];draw.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='red',width=2);draw.text((x*sx,y*sy),text,fill='red')
   for ps in v.get('landmarks',[]):
    for point in ps:
     if point.get('visibility') is not None and point['visibility']<.5:continue
     x,y=point['x']*view.width,point['y']*view.height;draw.ellipse((x-1,y-1,x+1,y+1),fill='lime')
   maskpath=None
   if 'map' in v:maskpath=root/'rust-eval'/model/(r['id']+'-mask.pgm')
   elif 'masks' in v:
    # Binary selfie exports foreground; multiclass/hair remain separate indexed maps.
    index=1 if len(v['masks'])==2 else 0
    maskpath=root/'reference-masks-v2'/model/v['masks'][index]['file']
    label.append('classes='+str(len(v['masks'])))
   if maskpath:
    with Image.open(maskpath) as im:
     mask=im.convert('L').resize(view.size)
    red=Image.new('RGB',view.size,'red');view=Image.blend(view,Image.composite(red,view,mask),.55)
   filename=r['id']+'-'+model+'.png';view.save(assets/filename)
   pages.append('<figure>'+html.escape(model)+'<br><img src="assets/'+filename+'"><figcaption>'+html.escape(' '.join(label))+'</figcaption></figure>')
  pages.append('</div></article>')
 (gallery/'index.html').write_text('\n'.join(pages))
 sheet=Image.new('RGB',(320*4,350*((len(contact)+3)//4)), '#dddddd')
 for i,card in enumerate(contact):sheet.paste(card,((i%4)*320,(i//4)*350))
 sheet.save(gallery/'contact.png');print(json.dumps({'sources':len(doc['images']),'model_outputs':len(models),'contact_images':len(contact)}))
if __name__=='__main__':main()
