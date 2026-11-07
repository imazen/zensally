#!/usr/bin/env python3
"""Partition a frozen manifest for independent bounded evaluation processes."""
import argparse,csv,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
with a.manifest.open() as f:
 r=csv.DictReader(f,delimiter='\t');fields=r.fieldnames;rows=list(r)
counts=[]
for shard in range(4):
 selected=[r for r in rows if int(r['rank'][-2:],16)%4==shard]
 with (a.output/f'shard-{shard}.tsv').open('x') as f:
  w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(selected)
 counts.append({'shard':shard,'natural_sources':len(selected),'stress_sources':sum(r['stress']=='true' for r in selected)})
(a.output/'source.json').write_text(json.dumps({'parent_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'counts':counts},indent=2)+'\n')
print(json.dumps(counts))
