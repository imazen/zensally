#!/usr/bin/env python3
"""Two bounded real-photo witnesses for continuous paired-head suitability."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
import torch
from PIL import Image
from artifact_identity import sha
from distill_attention_linear import write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','fits','resizer','encoder','output']:p.add_argument(name,type=Path)
    p.add_argument('--ridge-baseline',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    if sha(a.cache)!='3b85e5ed376b92d62d38e72fdf432d665b7aa62c944c6a874a9c83fc064773c4' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('source/pixel identity differs')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);source={s['id']:(i,s) for i,s in enumerate(c['sources'])};models=['geometry-only','aligned-distillation'];lookup={};events=[]
    if a.ridge_baseline is None:
        lookup={name:{(r['id'],r['query']):r for r in json.loads((a.fits/f'{name}-seed137/selected-predictions.json').read_text())} for name in models}
    else:
        if sha(a.ridge_baseline/'SHA256SUMS.json')!='318f88d1d3259433ead7d424f082a5ec63cb9908d1578e242968449286f910d5':raise ValueError('frozen ridge baseline differs')
        models=['geometry','person-all-pca8','person-largest-pca8'];lookup={name:{(r['id'],r['query']):r for r in json.loads(((a.ridge_baseline if name=='geometry' else a.fits)/(name+'-predictions.json')).read_text())} for name in models}
    for sid,(i,s) in source.items():
        if s['role']!='dev':continue
        for ri in range(9):
            if not c['target_valid'][i,ri].any():continue
            scale=torch.tensor([*s['native_size'],*s['native_size']]);pair=torch.tensor([lookup[models[-1]][sid,ri*2+m]['native'] for m in range(2)])/scale;error=float(((pair[None]-c['targets'][i,ri]).abs()*128).amax((-1,-2)).masked_fill(~c['target_valid'][i,ri],torch.inf).min());events.append((error,sid,ri))
    if a.ridge_baseline is None:
        worst=max(events,key=lambda r:(r[0],r[1],r[2]));requests=[('fixed counter3:1 face/story witness','openimages:9b15ccd240edae57',c['ratios'].index('3:1')),('largest paired max-edge error on known DEV reference',worst[1],worst[2])]
    else:
        excluded={'openimages:9b15ccd240edae57','openimages:48cd9b5fa18a01d8','openimages:9fa07d98319d63b9','openimages:6f66b42a0250bed6'};changes=[]
        for key,before in lookup['geometry'].items():
            after=lookup['person-largest-pca8'][key];sid,q=key
            if before['role']!='dev' or q>=18 or not before['head_known'] or not before['reference_known'] or sid in excluded:continue
            s=source[sid][1];scale=[*s['native_size'],*s['native_size']];movement=max(abs(x-y)*128/z for x,y,z in zip(before['native'],after['native'],scale));changes.append(dict(id=sid,ratio_index=q//2,query=q,movement=movement,before=before['weak_cut'],after=after['weak_cut']))
        repair=max([r for r in changes if r['before'] and not r['after']],key=lambda r:(r['movement'],r['id'],r['query']));introduced=max([r for r in changes if not r['before'] and r['after'] and r['id']!=repair['id']],key=lambda r:(r['movement'],r['id'],r['query']));requests=[('largest movement weak-head repair on a fresh source',repair['id'],repair['ratio_index']),('largest movement introduced weak-head cut on another fresh source',introduced['id'],introduced['ratio_index'])]
    a.output.mkdir(parents=True,exist_ok=False);cases=[];articles=[]
    for number,(reason,sid,ri) in enumerate(requests):
        i,s=source[sid];w,h=s['working_size'];rawpath=Path(s['png']).with_suffix('.rgb');raw=rawpath.read_bytes()
        if sha(rawpath)!=s['rgb_sha256'] or sha(Path(s['png']))!=s['png_sha256']:raise ValueError('source pixels changed')
        png=a.output/f'case{number}-source.png';shutil.copy2(s['png'],png)
        with Image.open(png) as im:
            im.load()
            if im.size!=(w,h) or im.convert('RGB').tobytes()!=raw:raise ValueError('source readback differs')
        ratio=c['ratios'][ri];panes=[dict(name='source',path=str(png),sha256=sha(png))];figures=[f'<figure><img src="{png.name}"><figcaption>Full source</figcaption></figure>'];boxes=[]
        for model in models:
            for m in range(2):boxes.append((model+('-minimal' if m==0 else '-maximal'),lookup[model][sid,ri*2+m]['native']))
        if c['target_valid'][i,ri].any():
            k=int(c['target_valid'][i,ri].nonzero()[0]);scale=torch.tensor([*s['native_size'],*s['native_size']]);pair=(c['targets'][i,ri,k]*scale).round().long().tolist()
            for m in range(2):boxes.append(('cached-weak-reference'+('-minimal' if m==0 else '-maximal'),pair[m]))
        for name,box in boxes:
            result=render(raw,w,h,w*3,s['native_size'],box,ratio,a.output/f'case{number}-{name}',a.resizer,a.encoder,unit=384//max(map(int,ratio.split(':'))));panes.append(dict(name=name,native_box=box,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{name}</figcaption></figure>')
        cases.append(dict(id=sid,ratio=ratio,reason=reason,source=s['source'],panes=panes));articles.append(f'<article><h2>{html.escape(sid)} {ratio}</h2><p>{reason}</p><a href="{html.escape(s["source"]["source_url"],quote=True)}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in a.output.iterdir() if p.is_file())
    if size>16*1024*1024:raise ValueError('bounded witness allowance exceeded')
    caption='Seed137 controls.' if a.ridge_baseline is None else 'FIT-family-selected automatic person-region controls; four previously inspected sources excluded only from diagnostic witness selection.'
    write(a.output/'summary.json',dict(cases=cases,asset_bytes=size,api_calls=0,test_inference=False,model_promoted=False));(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Continuous nested head</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Continuous nested-head suitability</h1><p>'+caption+' Cached teacher rectangles are unreviewed weak references, not excellent or human-rated crops. This model has no contain head and is not promoted.</p>'+''.join(articles));print(json.dumps(dict(cases=[dict(id=c['id'],ratio=c['ratio']) for c in cases],panes=sum(len(c['panes']) for c in cases),asset_bytes=size)))


if __name__=='__main__':main()
