#!/usr/bin/env python3
"""Bounded actual-pixel nested-selector changes and explicit missing endpoints."""
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
    for name in ['audit','review','resizer','encoder']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.review/'SHA256SUMS.json')!='04c25c344c95b05eacded621b45b104eb73ba0bbfbaf092d98eda71088df1e84':raise ValueError('fixed FIT review differs')
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned pixel binaries differ')
    sources=json.loads((a.audit/'sources.json').read_text());source_map={(cohort,s['id']):s for cohort,values in sources.items() for s in values};pairs=json.loads((a.audit/'pairs.json').read_text());endpoints=json.loads((a.audit/'endpoints.json').read_text());lookup={(r['cohort'],r['id'],r['ratio_index'],r['policy']):r for r in pairs};raw={(r['cohort'],r['id'],r['query']):r for r in endpoints if r['policy']=='independent'}
    first=json.loads((a.review/'selection.json').read_text())['sources'][0]['id'];selected=[dict(cohort='new',id=first,ratio_index=8,reason='FIT performer: unavailable inner versus moved outer'),dict(cohort='new',id=first,ratio_index=7,reason='FIT performer: forced tighter inner')]
    old=[r for r in endpoints if r['cohort']=='old' and r['role']=='dev' and r['policy']=='outer-first' and r['essential'] and r['mode']=='maximal' and r['valid'] and r['offered']];winner=min(old,key=lambda r:(-(r['teacher_regret']-raw['old',r['id'],r['query']]['teacher_regret']),r['id'],r['query']));selected.append(dict(cohort='old',id=winner['id'],ratio_index=winner['query']//2,reason='largest old essential Maximal weak-regret increase'))
    new=[r for r in endpoints if r['cohort']=='new' and r['role']=='dev' and r['policy']=='outer-first' and r['essential'] and r['mode']=='maximal' and r['valid'] and r['offered'] and r['cut'] and not raw['new',r['id'],r['query']]['cut']]
    def movement(row):
        w,h=source_map['new',row['id']]['native_size'];return max(abs(x-y)/scale*128 for x,y,scale in zip(row['native_box'],raw['new',row['id'],row['query']]['native_box'],[w,h,w,h]))
    winner=min(new,key=lambda r:(-movement(r),r['id'],r['query']));selected.append(dict(cohort='new',id=winner['id'],ratio_index=winner['query']//2,reason='largest new essential added weak head cut by edge movement'))
    output=a.audit/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for number,request in enumerate(selected):
        source=source_map[request['cohort'],request['id']];w,h=source['working_size'];raw_path=Path(source.get('rgb',str(Path(source['png']).with_suffix('.rgb'))));pixels=raw_path.read_bytes()
        if sha(raw_path)!=source['rgb_sha256']:raise ValueError('source RGB differs')
        path=output/f'case{number}-source.png';shutil.copy2(source['png'],path)
        with Image.open(path) as im:
            im.load()
            if im.size!=(w,h) or im.convert('RGB').tobytes()!=pixels:raise ValueError('source PNG readback differs')
        panes=[dict(name='source',path=str(path),sha256=sha(path))];states=[];figures=[f'<figure><img src="{path.name}"><figcaption>Full source</figcaption></figure>'];ratio=lookup[request['cohort'],request['id'],request['ratio_index'],'independent']['ratio'];unit=768//max(map(int,ratio.split(':')))
        for policy in ['independent','outer-first','joint-sum','maximal-first']:
            row=lookup[request['cohort'],request['id'],request['ratio_index'],policy];states.append(row)
            for mode,box in [('minimal',row['outer_box']),('maximal',row['inner_box'])]:
                name=policy+'-'+mode
                if box is None:
                    figures.append(f'<figure><p>{name}: no strict inner candidate; no crop output</p></figure>');continue
                rendered=render(pixels,w,h,w*3,source['native_size'],box,ratio,output/f'case{number}-{name}',a.resizer,a.encoder,unit=unit);panes.append(dict(name=name,native_box=box,**rendered));figures.append(f'<figure><img src="{Path(rendered["path"]).name}"><figcaption>{name}; strict pair={row["strict_pair"]}</figcaption></figure>')
        cases.append(dict(**request,ratio=ratio,panes=panes,policies=states,source=source['source']));link=html.escape(source['source']['source_url'],quote=True);articles.append(f'<article><h2>{html.escape(request["id"])} {ratio}</h2><p>{request["reason"]}</p><a href="{link}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>64*1024*1024:raise ValueError('bounded witness storage exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,budget_bytes=64*1024*1024,api_calls=0,test_inference=False,training_eligible=False,model_promoted=False),indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Nested crop selection</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:420px;object-fit:contain}</style><h1>Nested selection witnesses</h1><p>Two fixed FIT performer requests and two deliberately selected DEV regressions. Missing inner crops have no output image. Actual owned pixels, independent exact RGB readback; no new labels or model promotion.</p>'+''.join(articles))
    print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),asset_bytes=size)))


if __name__=='__main__':main()
