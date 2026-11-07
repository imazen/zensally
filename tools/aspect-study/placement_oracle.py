#!/usr/bin/env python3
"""Independently exhaust every integer crop origin for the reported tall case."""
import argparse,csv,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('render',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
case='ILSVRC2012_test_00084919-canvas-1x8-p0';directory=a.render/'gallery'/case
maps={};results=[]
for r in csv.DictReader((a.render/'crops.tsv').open(),delimiter='\t'):
 if r['id']+'-'+r['case']!=case:continue
 method=r['method']
 if method not in maps:
  data=struct.unpack('<65536f',(directory/f'{method}-map.f32le').read_bytes())
  assert all(math.isfinite(v) and 0<=v<=1 for v in data)
  # Row projection is independent of the solver's sliding-column algorithm.
  mass=[sum(float(v)**2 for v in data[y*256:(y+1)*256]) for y in range(256)]
  prefix=[0.]
  for v in mass:prefix.append(prefix[-1]+v)
  maps[method]=prefix
 prefix=maps[method];sw,sh,x,y,w,h=(int(r[k]) for k in ['width','height','x','y','w','h'])
 assert x==0 and w==sw,'This independent oracle requires full-width crops'
 def at(pos):
  pos=max(0.,min(256.,pos));i=int(pos)
  return prefix[i] if i==256 else prefix[i]+(pos-i)*(prefix[i+1]-prefix[i])
 def score(top):return at((top+h)*256/sh)-at(top*256/sh)
 scores=[score(top) for top in range(sh-h+1)]
 actual=score(y);best=max(scores)
 assert abs(actual-best)<=max(1.,prefix[-1])*1e-10,(method,r['aspect'],r['mode'],actual,best)
 results.append({'method':method,'aspect':r['aspect'],'mode':r['mode'],'origin_y':y,'crop_height':h,'retained_squared_mass':actual,'best_squared_mass':best,'source_origins_checked':len(scores),'labelled_retention':float(r['retention'])})
assert len(results)==80
result={'case':case,'maps':len(maps),'crop_comparisons':len(results),'source_origins_checked':sum(r['source_origins_checked'] for r in results),'rows':results}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
