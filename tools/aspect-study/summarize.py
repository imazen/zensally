#!/usr/bin/env python3
"""Aggregate complete fixed studies; select policies only on calibration sources."""
import argparse,csv,hashlib,json,math,random,statistics
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--manifest',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
with a.manifest.open() as f:source_families={r['id']:r['image_sha256'] for r in csv.DictReader(f,delimiter='\t')}
methods=['stretch','letterbox','letterbox-mean','quadrants','overlap-refine','aspect-refine']

def aspect_bin(w,h):
 r=max(w/h,h/w)
 return '1-1.5' if r<1.5 else '1.5-2' if r<2 else '2-4' if r<4 else '4+'

cases={};crop_records={}
for kind,expected in [('natural',5019),('stress',128*21)]:
 files=[a.run/f'{kind}-{i}' for i in range(4)]
 assert all((d/'complete.txt').exists() for d in files),f'incomplete {kind}'
 tables=[]
 for d in files:
  t=pcsv.read_csv(d/'metrics.tsv',parse_options=pcsv.ParseOptions(delimiter='\t'))
  tables.append(t)
  for r in t.to_pylist():
   key=(kind,r['id'],r['case'],r['method']);assert key not in cases
   assert all(math.isfinite(float(r[k])) for k in ['mae','iou','f_beta','retention_mean','regret_mean','retention_min'])
   r.update(kind=kind,bin=aspect_bin(r['width'],r['height']))
   cases[key]=r
 pq.write_table(pa.concat_tables(tables),a.output/f'{kind}-metrics.parquet',compression='zstd')
 assert len([k for k in cases if k[0]==kind])==expected*6
 crop_tables=[]
 for d in files:
  t=pcsv.read_csv(d/'crops.tsv',parse_options=pcsv.ParseOptions(delimiter='\t'))
  crop_tables.append(t)
  with (d/'crops.tsv').open() as f:
   for r in csv.DictReader(f,delimiter='\t'):
    key=(kind,r['id'],r['case'],r['method'],r['mode'])
    c=crop_records.setdefault(key,{'retention':0.,'regret':0.,'area':0.,'count':0})
    for field in ['retention','regret']:c[field]+=float(r[field])
    c['area']+=int(r['w'])*int(r['h'])/(int(r['width'])*int(r['height']));c['count']+=1
 pq.write_table(pa.concat_tables(crop_tables),a.output/f'{kind}-crops.parquet',compression='zstd')
 for k in crop_records:
  if k[0]==kind:
   assert crop_records[k]['count']==5

for key,r in cases.items():
 for mode in ['minimal','maximal']:
  c=crop_records[(*key,mode)]
  for field in ['retention','regret','area']:r[f'{mode}_{field}']=c[field]/c['count']

# Freeze selection before any evaluation aggregation; this branch reads only
# calibration rows. A source counts once when checking bin admission.
policy={}
for kind in ['natural','stress']:
 for bin_ in ['1-1.5','1.5-2','2-4','4+']:
  rows=[r for r in cases.values() if r['kind']==kind and r['split']=='calibration' and r['bin']==bin_]
  sources={r['id'] for r in rows}
  choice='stretch'
  if len(sources)>=30:
   scores={m:statistics.mean(r['minimal_regret'] for r in rows if r['method']==m) for m in methods}
   best=min(scores.values());eligible=[m for m in methods if scores[m]<=best+1e-6]
   choice=min(eligible,key=lambda m:(statistics.mean(r['passes'] for r in rows if r['method']==m),m))
  policy[f'{kind}/{bin_}']={'method':choice,'calibration_sources':len(sources)}
(a.output/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')

fields=['mae','iou','f_beta','passes','minimal_retention','minimal_regret','minimal_area','maximal_retention','maximal_regret','maximal_area']
# Average variants within each source before uncertainty calculations.
groups=defaultdict(lambda:defaultdict(list))
for r in cases.values():
 for bin_ in [r['bin'],'all']:
  groups[(r['kind'],r['split'],bin_,r['method'])][r['id']].append(r)
 if r['method']==policy[f"{r['kind']}/{r['bin']}"]['method']:
  for bin_ in [r['bin'],'all']:groups[(r['kind'],r['split'],bin_,'policy')][r['id']].append(r)

source_means={}
rows=[]
for key,values in sorted(groups.items()):
 means={ident:{field:statistics.mean(r[field] for r in rs) for field in fields} for ident,rs in values.items()}
 source_means[key]=means
 row=dict(zip(['kind','split','bin','method'],key));row['sources']=len(values);row['cases']=sum(map(len,values.values()))
 row.update({field:statistics.mean(v[field] for v in means.values()) for field in fields});rows.append(row)

paired=[]
for key,means in sorted(source_means.items()):
 kind,split,bin_,method=key
 if split!='evaluation' or method=='stretch' or (kind=='natural' and bin_!='all'):continue
 baseline=source_means[(kind,split,bin_,'stretch')];ids=sorted(means);assert set(ids)==set(baseline)
 delta={f:[means[i][f]-baseline[i][f] for i in ids] for f in ['mae','minimal_regret']}
 family_indices=defaultdict(list)
 for index,ident in enumerate(ids):family_indices[source_families[ident]].append(index)
 clusters=list(family_indices.values())
 rng=random.Random(20261107);draws={f:[] for f in delta}
 cluster_sums={f:[sum(values[i] for i in c) for c in clusters] for f,values in delta.items()}
 for _ in range(2000):
  chosen=rng.choices(range(len(clusters)),k=len(clusters));weight=sum(len(clusters[i]) for i in chosen)
  for field,values in cluster_sums.items():draws[field].append(sum(values[i] for i in chosen)/weight)
 result={'kind':kind,'bin':bin_,'method':method,'sources':len(ids),'independent_byte_clusters':len(clusters),'differences':{}}
 for field,values in delta.items():
  samples=sorted(draws[field]);result['differences'][field]={'mean':statistics.mean(values),'ci95':[samples[49],samples[1949]]}
 paired.append(result)

# Preserve large typed tables, plus compact readable data and all fixed arms.
(a.output/'summary.json').write_text(json.dumps({'schema_version':1,'methods':methods,'bootstrap_replicates':2000,'bootstrap_seed':20261107,'grid':256,'source_cases':{'natural':5019,'stress':2688},'rows':rows,'paired':paired,'policy':policy},indent=2)+'\n')
with (a.output/'summary.tsv').open('x') as f:
 w=csv.DictWriter(f,list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
# Deterministic examples and biggest paired crop-regret improvements/regressions.
selected={}
for kind in ['natural','stress']:
 options=[]
 for key,r in cases.items():
  if key[0]!=kind or r['split']!='evaluation' or r['method']=='stretch':continue
  base=cases[(kind,r['id'],r['case'],'stretch')]
  options.append((r['minimal_regret']-base['minimal_regret'],r['id'],r['case'],r['method']))
 for label,values in [('improvement',sorted(options)),('regression',sorted(options,reverse=True))]:
  taken=0
  for delta,ident,case,method in values:
   k=(kind,ident,case)
   if k in selected:continue
   selected[k]={'kind':kind,'id':ident,'case':case,'selection':label,'method':method,'minimal_regret_delta':delta};taken+=1
   if taken==5:break
 # Add aspect-stratified examples, independent of method quality.
 for bin_ in ['1-1.5','1.5-2','2-4','4+']:
  choices=sorted((r for r in cases.values() if r['kind']==kind and r['split']=='evaluation' and r['method']=='stretch' and r['bin']==bin_),key=lambda r:hashlib.sha256((r['id']+r['case']).encode()).hexdigest())
  if choices:
   r=choices[0];selected.setdefault((kind,r['id'],r['case']),{'kind':kind,'id':r['id'],'case':r['case'],'selection':'aspect-stratified'})
selection=list(selected.values())
(a.output/'gallery-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
for kind in ['natural','stress']:
 (a.output/f'gallery-{kind}.txt').write_text(''.join(r['id']+'-'+r['case']+'\n' for r in selection if r['kind']==kind))
print(json.dumps({'summary_rows':len(rows),'paired_comparisons':len(paired),'selected_gallery_cases':len(selection),'policy':policy}))
