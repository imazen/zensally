#!/usr/bin/env python3
"""Create source-preserving selected-gallery manifests; never alter split roles."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
with a.manifest.open() as f:r=csv.DictReader(f,delimiter='\t');fields=r.fieldnames;rows=list(r)
selected=json.loads((a.analysis/'gallery-selection.json').read_text())
for kind in ['natural','stress']:
 ids={r['id'] for r in selected if r['kind']==kind}
 with (a.output/f'{kind}.tsv').open('x') as f:
  w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(r for r in rows if r['id'] in ids)
print('selected source manifests written')
