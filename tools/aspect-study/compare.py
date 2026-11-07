#!/usr/bin/env python3
"""Require unchanged predictions and sizes before comparing repaired crop placement."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('original',type=Path);p.add_argument('corrected',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
old_methods={'stretch','letterbox','letterbox-mean','quadrants','overlap-refine','aspect-refine'}
new_methods=old_methods|{'letterbox-blur','letterbox-edge-blur'}
metric_count=crop_count=changed=0
groups=defaultdict(lambda:{'n':0,'old_retention':0.,'new_retention':0.})
reported=[]
for kind in ['natural','stress']:
 for shard in range(4):
  old=a.original/f'{kind}-{shard}';new=a.corrected/f'{kind}-{shard}'
  assert (old/'complete.txt').exists() and (new/'complete.txt').exists()
  assert (old/'mask-hashes.tsv').read_bytes()==(new/'mask-hashes.tsv').read_bytes()
  def key(r):return (r['id'],r['case'],r['method'])
  original={key(r):r for r in csv.DictReader((old/'metrics.tsv').open(),delimiter='\t')}
  seen=set()
  for r in csv.DictReader((new/'metrics.tsv').open(),delimiter='\t'):
   assert r['method'] in new_methods
   if r['method'] not in old_methods:continue
   k=key(r);assert k not in seen;seen.add(k);before=original[k]
   for field in ['split','width','height','passes','mae','iou','f_beta']:
    assert r[field]==before[field],(kind,shard,k,field,before[field],r[field])
   metric_count+=1
  assert seen==set(original)
  def crop_key(r):return (*key(r),r['aspect'],r['mode'])
  original={crop_key(r):r for r in csv.DictReader((old/'crops.tsv').open(),delimiter='\t')}
  seen=set()
  for r in csv.DictReader((new/'crops.tsv').open(),delimiter='\t'):
   x,y,w,h=(int(r[f]) for f in ['x','y','w','h'])
   assert x>=0 and y>=0 and w>0 and h>0 and x+w<=int(r['width']) and y+h<=int(r['height'])
   if r['method'] not in old_methods:continue
   k=crop_key(r);assert k not in seen;seen.add(k);before=original[k]
   for field in ['split','width','height','w','h']:assert r[field]==before[field],(kind,k,field)
   crop_count+=1;changed+=any(r[f]!=before[f] for f in ['x','y'])
   g=groups[(kind,r['split'],r['method'],r['mode'])]
   g['n']+=1;g['old_retention']+=float(before['retention']);g['new_retention']+=float(r['retention'])
   if r['id']=='ILSVRC2012_test_00084919' and r['case']=='canvas-1x8-p0' and r['method']=='stretch' and r['aspect']=='9:16':
    reported.append({'mode':r['mode'],'old':before,'corrected':r})
  assert seen==set(original)
assert metric_count==46242 and crop_count==462420 and len(reported)==2
rows=[]
for key,g in sorted(groups.items()):
 n=g['n'];rows.append(dict(zip(['kind','split','method','mode'],key),count=n,old_retention=g['old_retention']/n,new_retention=g['new_retention']/n))
result={'unchanged_prediction_rows':metric_count,'unchanged_crop_sizes':crop_count,'changed_crop_origins':changed,'reported_case':reported,'rows':rows}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
