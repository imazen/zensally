#!/usr/bin/env python3
"""Preserve bounded full-source witnesses of lost weak-head corroboration."""
import argparse
import html
import json
from pathlib import Path
import shutil
from PIL import Image
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','manifest']:p.add_argument(name,type=Path)
    a=p.parse_args();rows=json.loads((a.audit/'head-coverage.json').read_text())
    rows=[r for r in rows if r['role']=='dev' and r['threshold']==.1]
    lookup={(r['id'],r['rule']):r for r in rows};sources={r['id']:r for r in json.loads(a.manifest.read_text())['images']};links=json.loads((a.audit/'source-links.json').read_text());selected=[]
    for rule,limit in [('person',3),('other-family',1)]:
        pool=sorted([r for r in rows if r['rule']==rule and lookup[r['id'],'all']['covered_centers']>r['covered_centers']],key=lambda r:(-(lookup[r['id'],'all']['covered_centers']-r['covered_centers']),r['id']))
        added=0
        for r in pool:
            if r['id'] in selected:continue
            selected.append(r['id']);added+=1
            if added==limit:break
    if len(selected)!=4:raise ValueError('registered four loss witnesses required')
    output=a.audit/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for index,sid in enumerate(selected):
        source=sources[sid];path=Path(source['display']);raw=Path(source['rgb'])
        if sha(raw)!=source['rgb_sha256']:raise ValueError('owned witness RGB differs')
        dest=output/f'case{index}-source.png';shutil.copy2(path,dest)
        with Image.open(dest) as image:
            image.load()
            if image.size!=(source['width'],source['height']) or image.convert('RGB').tobytes()!=raw.read_bytes():raise ValueError('source PNG readback differs')
        counts={rule:lookup[sid,rule]['covered_centers'] for rule in ['all','other-family','person','both','either']}
        cases.append(dict(id=sid,panes=[dict(name='owned-source',path=str(dest),sha256=sha(dest))],counts=counts,source=links[sid]))
        link=html.escape(links[sid]['source_url'],quote=True)
        articles.append(f'<article><h2>{html.escape(sid)}</h2><a href="{link}">Original source</a><p>Weak head centers covered: {html.escape(str(counts))}</p><img src="{dest.name}"></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>16*1024*1024:raise ValueError('bounded source witness storage exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,budget_bytes=16*1024*1024,api_calls=0,training_eligible=False,model_promoted=False),indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Corroboration coverage losses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}img{max-width:100%;max-height:640px}article{border-top:1px solid #999;margin-top:24px}</style><h1>Corroboration coverage losses</h1><p>Four deliberately selected DEV source witnesses. Weak head nominations include backs/occlusions and do not establish actual-face recall. Unsupported proposals remain uncertain; none is discarded. No new training labels.</p>'+''.join(articles))
    print(json.dumps(dict(sources=len(cases),asset_bytes=size)))


if __name__=='__main__':main()
