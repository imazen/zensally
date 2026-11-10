#!/usr/bin/env python3
"""Render audited subject-choice rectangles with the owned crop renderer."""
import argparse
import csv
import hashlib
import html
import json
import os
import subprocess
from pathlib import Path
from expanded import HEADER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['audit','sources','renderer','resizer','output']:p.add_argument(key,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    pieces=[];readbacks=[]
    for index,item in enumerate(json.loads(a.audit.read_text())):
        identity=item['id'];name=f'{identity}-{item["teacher"]}';w,h=item['width'],item['height']
        zones=json.loads(Path(item['zones']).read_text());audit=json.loads(Path(item['audit']).read_text())
        rows=[]
        for row in audit['rows']:
            rw,rh=map(int,row['ratio'].split(':'));x,y,right,bottom=row['projected_xyxy']
            filename=f'{rw}x{rh}-{row["mode"]}-{row["option"]}.png'
            row['file']=filename
            rows.append([name,name,row['mode'],rw,rh,w,h,'subject-choice',x,y,right-x,bottom-y,filename,'audited-exact-aspect'])
        table=a.output/(name+'.tsv')
        with table.open('x') as stream:
            writer=csv.writer(stream,delimiter='\t',lineterminator='\n');writer.writerow(HEADER);writer.writerows(rows)
        subprocess.run([str(a.renderer),str(a.sources/(identity+'.png')),str(a.output/name),name,str(table),'768'],check=True,
            env=dict(os.environ,ZENSALLY_PREVIEW_RESIZER=str(a.resizer)))
        pieces.append(f'<section><h2>{html.escape(name)}</h2><img class="original" src="{name}/source.png"><div class="grid">')
        for row in audit['rows']:
            check=row['projected'];bad=not check['geometry_compatible']
            pieces.append(f'<figure class="{"bad" if bad else "review"}"><img src="{name}/{row["file"]}"><figcaption>{row["ratio"]} {row["mode"]} option{row["option"]+1}<br>{"GEOMETRY REJECT" if bad else "Needs visual review"}<br>Cut face IDs: {html.escape(str(check["cut_faces"]))}<br>xyxy: {row["projected_xyxy"]}</figcaption></figure>')
        pieces.append('</div></section>')
        for path in sorted((a.output/name).glob('*.png')):
            with Image.open(path) as image:
                image.load();readbacks.append(dict(file=str(path.relative_to(a.output)),size=list(image.size),rgb_sha256=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),png_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Subject-choice pilot</title><style>body{font:16px system-ui;background:#181a20;color:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}img{width:100%;height:260px;object-fit:contain}.original{height:450px}figure{margin:0;padding:8px;border:2px solid #899}.bad{border-color:#f66}figcaption{font-size:14px}section{border-top:1px solid #999;margin-top:30px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>Subject-choice pilot</h1><p>Actual owned-codec crop pixels at exact integer aspect. Sources may choose one person or subgroup. Red border: geometry rejection against supplied face ROIs. Other windows remain unreviewed proposals; neither teacher supplied complete certified face coverage. Contain requests retain the original and are omitted from this crop-only gallery. No labels are training-admitted.</p>'''+''.join(pieces)
    (a.output/'index.html').write_text(page)
    (a.output/'readbacks.json').write_text(json.dumps(readbacks,indent=2)+'\n')
    print(json.dumps(dict(sources=len(json.loads(a.audit.read_text())),crop_options=sum(len(json.loads(Path(i['audit']).read_text())['rows']) for i in json.loads(a.audit.read_text())),png_readbacks=len(readbacks))))


if __name__=='__main__':main()
