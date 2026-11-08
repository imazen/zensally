#!/usr/bin/env python3
"""Storage comparison: owned PNG/JXL codecs, independent lossless pixel oracle."""
import argparse,hashlib,json,math,struct,subprocess
from pathlib import Path
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('source',type=Path);p.add_argument('output',type=Path)
 p.add_argument('--png-reader',required=True);p.add_argument('--encoder',required=True)
 p.add_argument('--decoder',required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for src in sorted(a.source.glob('*.png')):
  data=src.read_bytes();i=8
  while i<len(data):
   n=struct.unpack('>I',data[i:i+4])[0];kind=data[i+4:i+8]
   if kind in (b'iCCP',b'cHRM',b'gAMA',b'cICP'):
    raise ValueError('explicit color metadata requires a preservation policy: '+str(src))
   i+=n+12
  work=a.output/src.stem;work.mkdir();raw=work/'source.rgb8'
  dims=json.loads(subprocess.check_output([a.png_reader,str(src),str(raw)]))
  with Image.open(src) as image:
   image.load()
   if image.mode!='RGB' or image.size!=(dims['width'],dims['height']) or image.tobytes()!=raw.read_bytes():
    raise ValueError('owned PNG decode differs from independent oracle')
  ref=raw.read_bytes();row={'id':src.stem,'png_bytes':src.stat().st_size,'png_sha256':sha(src),'rgb_sha256':sha(raw),**dims,'variants':{}}
  for mode in ['lossless','0.5']:
   out=work/(mode+'.jxl')
   subprocess.run([a.encoder,str(raw),str(dims['width']),str(dims['height']),str(out),mode],check=True)
   decoded=work/(mode+'.rgb8')
   subprocess.run([a.decoder,str(out),str(decoded)],check=True)
   oracle=work/(mode+'-oracle.png')
   subprocess.run(['djxl',str(out),str(oracle),'--num_threads=2'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   ours=decoded.read_bytes()
   with Image.open(oracle) as image:
    image.load();independent=image.convert('RGB').tobytes()
    if image.size!=(dims['width'],dims['height']):raise ValueError('oracle dimensions differ')
   if mode=='lossless' and (ours!=ref or independent!=ref):raise ValueError('lossless pixels differ')
   if len(ours)!=len(ref):raise ValueError('decoded RGB length differs')
   square=sum((x-y)**2 for x,y in zip(ref,ours));mse=square/len(ref)
   row['variants'][mode]={'bytes':out.stat().st_size,'sha256':sha(out),'decoded_rgb_sha256':sha(decoded),
      'lossless_exact':ours==ref,'independent_exact':independent==ref,
      'decoder_max_difference':max(abs(x-y) for x,y in zip(ours,independent)),
      'max_sample_error':max(abs(x-y) for x,y in zip(ref,ours)),
      'psnr_db':10*math.log10(255**2/mse) if mse else None}
  rows.append(row)
  with (a.output/'progress.json').open('w') as f:json.dump(rows,f,indent=2)
  print(json.dumps(row),flush=True)
 totals={'images':len(rows),'png_bytes':sum(r['png_bytes'] for r in rows),
   'variant_bytes':{m:sum(r['variants'][m]['bytes'] for r in rows) for m in ['lossless','0.5']},'records':rows}
 with (a.output/'summary.json').open('x') as f:json.dump(totals,f,indent=2)
 print(json.dumps({k:v for k,v in totals.items() if k!='records'}))
if __name__=='__main__':main()
