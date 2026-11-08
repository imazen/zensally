#!/usr/bin/env python3
"""Acquire pinned JXL bytes; use owned decode, encode and resize processes."""
import argparse,csv,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
from corpus import save,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('output',type=Path);p.add_argument('decoder',type=Path);p.add_argument('renderer',type=Path);p.add_argument('resizer',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);sources=json.loads((a.corpus/'sources.json').read_text());protocol=json.loads((a.corpus/'protocol.json').read_text());records=[]
    for s in sources:
        if s['use']=='test':continue # frozen test membership is not decoded/annotated now
        directory=a.output/s['id'];directory.mkdir(exist_ok=True);finished=directory/'source.json'
        if finished.exists():
            record=json.loads(finished.read_text())
            for key,name in [('sha256','source.jxl'),('png_sha256','source.png'),('preview_sha256','preview.png'),('tensor_sha256','tensor.rgb')]:
                if sha((directory/name).read_bytes())!=record[key]:raise ValueError('retained acquisition bytes changed')
            records.append(record);continue
        url=f'https://raw.githubusercontent.com/imazen/imazen-26-jxl-srgb/{protocol["export_commits"][s["bucket"]]}/'+quote(s['path'])
        jxl=directory/'source.jxl'
        if not jxl.exists():
            with urlopen(Request(url,headers={'User-Agent':'zensally-crop-research'}),timeout=120) as response:data=response.read()
            if sha(data)!=s['sha256']:raise ValueError('canonical JXL hash mismatch')
            jxl.write_bytes(data)
        if sha(jxl.read_bytes())!=s['sha256']:raise ValueError('canonical JXL hash mismatch')
        raw=directory/'source.rgb';decoded=directory/'decode.json'
        if not decoded.exists():
            if raw.exists():raise ValueError('incomplete decode retained; inspect before replacing')
            result=json.loads(subprocess.run([str(a.decoder),str(jxl),str(raw)],check=True,capture_output=True,text=True).stdout);save(decoded,result)
        result=json.loads(decoded.read_text());w,h=result['width'],result['height']
        if (w,h)!=(int(s['width']),int(s['height'])):raise ValueError('decoded/manifest dimensions disagree')
        profile=result['profile']
        simple_srgb='transfer_function: SRGB' in profile and ('primaries: SRGB' in profile or 'Grayscale' in profile)
        # The pinned corpus classifies embedded ICC sources too. Do not infer
        # transfer/primaries from an ICC description string or its filename.
        declared_icc_srgb=profile.startswith('Icc(') and s['colour'] in ['srgb','assumed-srgb']
        if not (simple_srgb or declared_icc_srgb):raise ValueError('source requires an explicit color conversion policy')
        png=directory/'source.png'
        if not png.exists():subprocess.run([str(a.renderer),'--from-rgb',str(w),str(h),str(raw),str(png)],check=True)
        rgb=raw.read_bytes()
        if len(rgb)!=w*h*3:raise ValueError('RGB8 size mismatch')
        # Aspect-preserving model input. Padding is outside valid_rect, never stretch.
        tw,th=max(1,round(w*256/max(w,h))),max(1,round(h*256/max(w,h)))
        resized=subprocess.run([str(a.resizer),str(w),str(h),str(w*3),f'0,0,{w},{h},{tw},{th}'],input=rgb,check=True,capture_output=True).stdout
        if len(resized)!=tw*th*3:raise ValueError('resizer output size mismatch')
        tensor=bytearray(256*256*3);ox,oy=(256-tw)//2,(256-th)//2
        for y in range(th):tensor[((oy+y)*256+ox)*3:((oy+y)*256+ox+tw)*3]=resized[y*tw*3:(y+1)*tw*3]
        (directory/'tensor.rgb').write_bytes(tensor)
        pw,ph=max(1,round(w*768/max(w,h))),max(1,round(h*768/max(w,h)))
        preview_raw=directory/'preview.rgb'
        preview_raw.write_bytes(subprocess.run([str(a.resizer),str(w),str(h),str(w*3),f'0,0,{w},{h},{pw},{ph}'],input=rgb,check=True,capture_output=True).stdout)
        preview=directory/'preview.png'
        if not preview.exists():subprocess.run([str(a.renderer),'--from-rgb',str(pw),str(ph),str(preview_raw),str(preview)],check=True)
        record={**s,'width':w,'height':h,'directory':str(directory),'canonical_url':url,'png_sha256':sha(png.read_bytes()),'rgb_sha256':sha(rgb),'preview_sha256':sha(preview.read_bytes()),'tensor_sha256':sha(tensor),'valid_rect':[ox/256,oy/256,(ox+tw)/256,(oy+th)/256],'decoded_profile':profile,'training_eligible':False,'human_rated':False}
        save(finished,record);records.append(record)
        (a.output/'progress.json').write_text(json.dumps({'stage':'acquiring','pid':os.getpid(),'completed':len(records),'required':len(sources)-sum(s['use']=='test' for s in sources),'last_source':s['id'],'checked_at':datetime.now(timezone.utc).isoformat()}))
        print(json.dumps({'acquired':s['id'],'use':s['use'],'completed':len(records)}),flush=True)
    save(a.output/'sources.json',records)
if __name__=='__main__':main()
