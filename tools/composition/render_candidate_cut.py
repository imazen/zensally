#!/usr/bin/env python3
"""Actual-pixel cut-risk repair, retained cut and explicit abstention witnesses."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
import torch
from PIL import Image
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','cache','metadata','resizer','encoder']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    expected=[(a.cache,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.resizer,'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041'),(a.encoder,'626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('owned pixel inputs differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);sources={s['id']:s for s in json.loads(a.metadata.read_text())['new']};index={s['id']:i for i,s in enumerate(c['sources'])};rows=json.loads((a.fits/'routing/predictions.json').read_text());lookup={(r['model'],r['id'],r['query']):r for r in rows if r['policy']=='corrected'};real='hidden-attention-seed31';models=['unfiltered',real,'permuted-attention-seed31','geometry-seed31'];accepted=[r for r in rows if r['policy']=='corrected' and r['model']==real and r['essential'] and r['reference_known'] and r['head_known']]
    def movement(row):
        original=lookup['unfiltered',row['id'],row['query']];i=index[row['id']];return float(((c['query_boxes'][i,row['query'],row['selected']]-c['query_boxes'][i,row['query'],original['selected']]).abs()*128).max())
    repair=min([r for r in accepted if r['offered'] and not r['weak_cut'] and lookup['unfiltered',r['id'],r['query']]['weak_cut']],key=lambda r:(-movement(r),r['id'],r['query']));remaining=min([r for r in accepted if r['offered'] and r['weak_cut']],key=lambda r:(r['query']//2 not in [7,8],r['id'],r['query']));abstain=min([r for r in accepted if not r['offered']],key=lambda r:(r['query']//2 not in [7,8],r['id'],r['query']));requests=[dict(id=r['id'],ratio_index=r['query']//2,reason=reason) for r,reason in [(repair,'largest normalized movement repairing a weak head cut on an offered endpoint'),(remaining,'remaining weak head cut; prioritize3:1/1:3'),(abstain,'accepted reference now unavailable; prioritize3:1/1:3')]]
    output=a.fits/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for number,request in enumerate(requests):
        source=sources[request['id']];i=index[request['id']];w,h=source['working_size'];rawpath=Path(source.get('rgb',str(Path(source['png']).with_suffix('.rgb'))));pixels=rawpath.read_bytes()
        if sha(rawpath)!=source['rgb_sha256']:raise ValueError('source RGB differs')
        path=output/f'case{number}-source.png';shutil.copy2(source['png'],path)
        with Image.open(path) as im:
            im.load()
            if im.size!=(w,h) or im.convert('RGB').tobytes()!=pixels:raise ValueError('source readback differs')
        panes=[dict(name='source',path=str(path),sha256=sha(path))];figures=[f'<figure><img src="{path.name}"><figcaption>Full source</figcaption></figure>'];ri=request['ratio_index'];ratio=c['ratios'][ri];nw,nh=source['native_size'];unit=256//max(map(int,ratio.split(':')));records=[]
        for model in models:
            for mode in range(2):
                q=ri*2+mode;row=lookup[model,request['id'],q];records.append(row);name=model+('-minimal' if mode==0 else '-maximal')
                if not row['offered']:
                    figures.append(f'<figure><p>{name}: unavailable pair; no crop output</p></figure>');continue
                native=(c['query_boxes'][i,q,row['selected']]*torch.tensor([nw,nh,nw,nh])).round().int().tolist();p,r=map(int,ratio.split(':'))
                if (native[2]-native[0])*r!=(native[3]-native[1])*p:raise ValueError('native aspect differs')
                result=render(pixels,w,h,w*3,source['native_size'],native,ratio,output/f'case{number}-{name}',a.resizer,a.encoder,unit=unit);panes.append(dict(name=name,record=row,native_box=native,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{name}; weak cut={row["weak_cut"]}</figcaption></figure>')
        cases.append(dict(**request,ratio=ratio,panes=panes,records=records,source=source['source']));url=html.escape(source['source']['source_url'],quote=True);articles.append(f'<article><h2>{html.escape(request["id"])} {ratio}</h2><p>{request["reason"]}</p><a href="{url}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>12*1024*1024:raise ValueError('bounded visual witness budget exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,api_calls=0,test_inference=False,training_eligible=False,model_promoted=False),indent=2)+'\n');(output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Candidate cut-risk witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Candidate cut-risk witnesses</h1><p>Three selected diagnostic requests, not representative quality ratings. Missing pairs have no output image. Weak head flags do not certify complete faces or artistic quality.</p>'+''.join(articles));print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),distinct_pngs=len({p['sha256'] for c in cases for p in c['panes']}),asset_bytes=size)))


if __name__=='__main__':main()
