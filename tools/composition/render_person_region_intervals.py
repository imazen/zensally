#!/usr/bin/env python3
"""Fresh repair/failure pixels for person-region continuous interval constraints."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
from PIL import Image
from artifact_identity import sha
from distill_attention_linear import write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['study','baseline','resizer','encoder','output']:p.add_argument(key,type=Path)
    a=p.parse_args()
    if sha(a.baseline/'SHA256SUMS.json')!='318f88d1d3259433ead7d424f082a5ec63cb9908d1578e242968449286f910d5' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('frozen parent/pixel tools differ')
    sources={r['id']:r['source'] for r in json.loads((a.study/'sources.json').read_text())}
    lookups={name:{(r['id'],r['query']):r for r in json.loads(path.read_text())} for name,path in [('parent',a.baseline/'geometry-predictions.json'),('upper',a.study/'geometry-upper/predictions.json'),('person',a.study/'geometry-person/predictions.json')]}
    excluded={'openimages:9b15ccd240edae57','openimages:48cd9b5fa18a01d8','openimages:9fa07d98319d63b9','openimages:6f66b42a0250bed6','openimages:a1d2f44a45a19cbd','openimages:1b8020d5a8d620a5'}
    eligible=[r for r in lookups['upper'].values() if r['role']=='dev' and r['query']<18 and r['head_known'] and r['offered'] and r['id'] not in excluded]
    repair=max([r for r in eligible if r['parent_cut'] and not r['weak_cut']],key=lambda r:(r['movement_grid'],r['id'],r['query']))
    remaining=max([r for r in eligible if r['weak_cut'] and r['id']!=repair['id']],key=lambda r:(r['movement_grid'],r['id'],r['query']))
    a.output.mkdir(parents=True,exist_ok=False);cases=[];articles=[]
    for index,(reason,chosen) in enumerate([('largest movement common-offer repair on a fresh source',repair),('largest movement remaining cut on another fresh source',remaining)]):
        sid=chosen['id'];s=sources[sid];q=chosen['query']//2*2;ratio=chosen['ratio'];w,h=s['working_size'];rawpath=Path(s['png']).with_suffix('.rgb');raw=rawpath.read_bytes()
        if sha(rawpath)!=s['rgb_sha256'] or sha(Path(s['png']))!=s['png_sha256']:raise ValueError('source pixels differ')
        source=a.output/f'case{index}-source.png';shutil.copyfile(s['png'],source)
        with Image.open(source) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=raw:raise ValueError('source independent readback differs')
        panes=[dict(name='source',path=str(source),sha256=sha(source))];figures=[f'<figure><img src="{source.name}"><figcaption>Whole source</figcaption></figure>'];missing=[]
        for name,rows in lookups.items():
            for m in range(2):
                row=rows[sid,q+m];label=name+('-minimal' if m==0 else '-maximal');box=row['native']
                if box is None:
                    missing.append(label);figures.append(f'<figure><p>No crop offered: {label}</p></figure>');continue
                result=render(raw,w,h,w*3,s['native_size'],box,ratio,a.output/f'case{index}-{label}',a.resizer,a.encoder,unit=256//max(map(int,ratio.split(':'))));panes.append(dict(name=label,native_box=box,weak_cut=row['weak_cut'],reference_known=row['reference_known'],**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{label}; nominated-head cut={row["weak_cut"]}</figcaption></figure>')
        cases.append(dict(id=sid,ratio=ratio,reason=reason,source=s['source'],panes=panes,missing=missing));articles.append(f'<article><h2>{html.escape(sid)} {ratio}</h2><p>{reason}</p><a href="{html.escape(s["source"]["source_url"],quote=True)}">Original source</a><div>{"".join(figures)}</div></article>')
    write(a.output/'summary.json',dict(cases=cases,api_calls=0,test_inference=False,model_promoted=False));(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Person-region interval witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:300px;object-fit:contain}</style><h1>Person-region interval constraints</h1><p>Frozen geometry parent, upper-body and full-person constraints. Diagnostic selections cover all nominated-head-known essential DEV requests; they are not excellent or human-rated labels. Other prior witness sources are excluded only from this visual selection.</p>'+''.join(articles))
    total=sum(p.stat().st_size for p in a.study.rglob('*') if p.is_file())
    if total>16*1024**2:raise ValueError('complete16MiB packet budget exceeded')
    print(json.dumps(dict(cases=[dict(id=c['id'],ratio=c['ratio']) for c in cases],panes=sum(len(c['panes']) for c in cases),total_study_bytes=total)))


if __name__=='__main__':main()
