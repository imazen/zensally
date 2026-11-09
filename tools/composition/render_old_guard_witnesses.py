#!/usr/bin/env python3
"""Render largest original-panel weak-regret changes using owned source pixels."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
from PIL import Image
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','resizer','encoder']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned pixel executables differ')
    sources={s['source_id']:s for s in json.loads((a.root/'manifest.json').read_text())['images']}
    rows=json.loads((a.root/'quality/predictions.json').read_text());lookup={(r['id'],r['query'],r['policy']):r for r in rows}
    pool=sorted([r for r in rows if r['policy']=='faces-partial-only' and r['query_valid'] and r['query']<18 and r['offered']],key=lambda r:(-(r['teacher_regret']-lookup[r['id'],r['query'],'raw']['teacher_regret']),r['id'],r['query']))
    selected=[];families=set()
    for r in pool:
        if r['family_id'] in families:continue
        families.add(r['family_id']);selected.append(r)
        if len(selected)==3:break
    output=a.root/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for number,r in enumerate(selected):
        source=sources[r['id']];raw=Path(source['rgb']).read_bytes()
        if sha(Path(source['rgb']))!=source['rgb_sha256']:raise ValueError('owned source changed')
        source_png=output/f'case{number}-source.png'
        if sha(Path(source['png']))!=source['png_sha256']:raise ValueError('source PNG changed')
        shutil.copy2(source['png'],source_png)
        with Image.open(source_png) as im:
            im.load()
            if im.size!=(source['width'],source['height']) or im.convert('RGB').tobytes()!=raw:raise ValueError('source PNG readback differs')
        panes=[dict(name='source',path=str(source_png),sha256=sha(source_png))]
        for name in ['raw','faces-partial-only','faces-pose-partial-only','contain']:
            record=lookup[r['id'],r['query'],name] if name!='contain' else None
            unit=768//max(map(int,r['ratio'].split(':')))
            asset=render(raw,source['width'],source['height'],source['width']*3,source['native_size'],record['native_box'] if record else None,r['ratio'],output/f'case{number}-{name}',a.resizer,a.encoder,unit=unit)
            panes.append(dict(name=name,record=record,**asset))
        delta=r['teacher_regret']-lookup[r['id'],r['query'],'raw']['teacher_regret']
        cases.append(dict(id=r['id'],query=r['query'],family_id=r['family_id'],ratio=r['ratio'],mode=r['mode'],weak_regret_increase=delta,panes=panes,source=source['source']))
        link=html.escape(source['source']['source_url'],quote=True)
        figures=''.join(f'<figure><img src="{Path(v["path"]).name}"><figcaption>{html.escape(v["name"])}</figcaption></figure>' for v in panes)
        articles.append(f'<article><h2>{html.escape(r["id"])} {r["ratio"]} {r["mode"]}</h2><p>Weak regret increase {delta}; diagnostic selection, not artistic truth.</p><a href="{link}">Original source</a><div>{figures}</div></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>64*1024*1024:raise ValueError('bounded witness storage exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,budget_bytes=64*1024*1024,api_calls=0,training_eligible=False,model_promoted=False),indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Original guard quality witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:420px;object-fit:contain}</style><h1>Original-panel filter changes</h1><p>Largest essential weak-regret increases, one request per source family. Actual owned pixels and independent exact PNG readback. No artistic labels admitted.</p>'+''.join(articles))
    print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),asset_bytes=size)))


if __name__=='__main__':main()
