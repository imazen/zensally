#!/usr/bin/env python3
"""Compact pose/face extraction, reusing exact-RGB cached reference runs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime-eval'))
from mediapipe_probe import make,load,summarize

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();doc=json.loads((a.root/'inputs.json').read_text());oldroot=Path('/mnt/v/output/zensally/runtime-models/2026-11-08');old=json.loads((oldroot/'manifest.json').read_text());oldimages={r['id']:r for r in old['images']};models={m['id']:m for m in old['models']};out=a.root/'geometry';out.mkdir(exist_ok=False);summaries={}
 for name,kind in [('float16-pose_landmarker_lite','pose'),('float16-blaze_face_full_range_sparse','faces')]:
  asset=models[name]
  if sha(Path(asset['path']))!=asset['sha256']:raise ValueError('model asset changed')
  cached=json.loads((oldroot/'reference-eval'/name/'results.json').read_text());cache={r['id']:r['result'] for r in cached['rows']};runner,call,_=make(asset);rows=[];reused=0
  try:
   for s in doc['images']:
    path=out/(s['id']+'-'+kind+'.json');key=s['id'] if s['synthetic'] else 'real-'+s['id'];prior=oldimages.get(key)
    exact=prior is not None and all(s[k]==prior[k] for k in ['width','height','rgb_sha256'])
    result=cache[key] if exact else summarize(call(load(s)),kind,s,out);reused+=int(exact)
    value={'id':s['id'],'role':s['role'],'family_id':s['family_id'],'rgb_sha256':s['rgb_sha256'],'width':s['width'],'height':s['height'],'model_sha256':asset['sha256'],'reused_exact_RGB':exact,'result':result}
    with path.open('x') as f:json.dump(value,f);f.write('\n')
    rows.append(s['id']);(out/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':kind,'completed':len(rows),'required':len(doc['images']),'checked_unix':time.time()}))
   summaries[kind]={'sources':len(rows),'cache_reused':reused,'computed':len(rows)-reused,'asset':asset};print(json.dumps({kind:summaries[kind]}),flush=True)
  finally:runner.close()
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
