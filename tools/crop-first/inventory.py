#!/usr/bin/env python3
"""Audit canonical source roles and family leakage without relabeling any image."""
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifests',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
roles={};splits={};hashes={}
for split in ['train','validate','test']:
 path=a.manifests/f'{split}.tsv';hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 with path.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
 counts=Counter();classes=Counter()
 for r in rows:
  assert r['split']==split and r['id'] not in roles
  roles[r['id']]=split;classes[r['content_class']]+=1
  w,h=int(r['width']),int(r['height']);assert w>0 and h>0
  ratio=max(w/h,h/w);counts['4+' if ratio>=4 else '2-4' if ratio>=2 else '<2']+=1
 # Report broad classes without copying household-linked content labels.
 splits[split]={'sources':len(rows),'aspect_bins':dict(counts),'content_groups':len(classes)}
family_path=a.manifests/'split_map_family.tsv';hashes[family_path.name]=hashlib.sha256(family_path.read_bytes()).hexdigest()
families=defaultdict(list)
with family_path.open() as f:
 for r in csv.DictReader(f,delimiter='\t'):
  if r['family']:families[r['family']].append(r['id'])
cross=[ids for ids in families.values() if len({roles[i] for i in ids})>1]
quarantine=sorted({i for ids in cross for i in ids if roles[i]=='train'})
result={'source_manifest_hashes':hashes,'splits':splits,'named_families':len(families),'cross_partition_named_families':len(cross),'training_sources_to_quarantine':len(quarantine),'training_quarantine_ids':quarantine,'canonical_roles_changed':False,'pixel_dimensions_checked':False,'note':'Dimensions are manifest observations; no new pixels downloaded. Existing named families are not a complete near-duplicate audit.'}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
