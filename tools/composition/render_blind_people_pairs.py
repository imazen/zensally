#!/usr/bin/env python3
"""Balanced blinded A/B nested pairs, with identity key hidden from review UI."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import secrets
import shutil
import sys
from PIL import Image
from artifact_identity import sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from crop_corridors import audit
from rendered_pixels import render

MODEL='hidden-regions-attention-seed137'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','routing','resizer','encoder','output']:p.add_argument(name,type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if sha(a.plan)!='1b2da79fe0a04d6a73121529d9e2c8940052cfb1812ccadcfc0fd70f01814e4a' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('fixed source/pixel contracts differ')
    state=json.loads((a.routing/'PROGRESS.json').read_text())
    if state['stage']!='complete' or state['completed']!=12:raise ValueError('all12 frozen policy outcomes required')
    sources=json.loads(a.plan.read_text())['sources'];rows=json.loads((a.routing/'predictions.json').read_text());lookup={(r['model'],r['applicability'],r['id'],r['query']):r for r in rows};ratio_q={r['ratio']:r['query']//2 for r in rows};inputs={str(a.plan):sha(a.plan),str(a.routing/'predictions.json'):sha(a.routing/'predictions.json')};requests=[]
    for i,s in enumerate(sources):
        for ratio in ['3:1','1:3']:
            ri=ratio_q[ratio];pairs={}
            for model,app in [('unfiltered','all'),(MODEL,'head')]:
                pair=[lookup[model,app,s['id'],ri*2+m]['native'] for m in range(2)]
                if not audit(pair[1],pair[0],*s['native_size'],ratio,[])['geometrically_admissible'] or pair[1][2]-pair[1][0]>=pair[0][2]-pair[0][0]:raise ValueError('strict nested matching-aspect pair required')
                pairs[model]=pair
            requests.append(dict(case_id=f'case{len(requests):02d}',source_index=i,id=s['id'],ratio=ratio,pairs=pairs))
    a.output.mkdir(parents=True,exist_ok=a.resume);reviews=a.output/'reviews'
    if reviews.exists() and any(reviews.glob('case*.json')):raise ValueError('do not regenerate pixels after any review')
    keypath=a.output/'_BLIND_KEY.json'
    if keypath.exists():
        if not a.resume:raise ValueError('explicit resume required')
        key=json.loads(keypath.read_text())
        if key['inputs']!=inputs or [r['case_id'] for r in key['cases']]!=[r['case_id'] for r in requests]:raise ValueError('blind source/model inputs changed')
    else:
        order=[False]*16+[True]*16;secrets.SystemRandom().shuffle(order);key=dict(inputs=inputs,cases=[dict(case_id=r['case_id'],id=r['id'],ratio=r['ratio'],models={'A':MODEL if flip else 'unfiltered','B':'unfiltered' if flip else MODEL},native_pairs=r['pairs']) for r,flip in zip(requests,order)]);write(keypath,key)
    keymap={r['case_id']:r for r in key['cases']};cases=[];articles=[]
    completed_path=a.output/'cases.json';prior=json.loads(completed_path.read_text())['cases'] if completed_path.exists() else [];done={r['case_id']:r for r in prior}
    for request in requests:
        cid=request['case_id'];s=sources[request['source_index']]
        if cid in done:
            case=done[cid]
            if any(sha(a.output/p['file'])!=p['sha256'] for p in case['panes']):raise ValueError('completed blind pixels changed')
            cases.append(case);continue
        w,h=s['working_size'];raw=Path(s['rgb']).read_bytes()
        if sha(Path(s['rgb']))!=s['rgb_sha256'] or sha(Path(s['png']))!=s['png_sha256']:raise ValueError('source pixels changed')
        source=a.output/f'source{request["source_index"]:02d}.png'
        if source.exists():
            if sha(source)!=s['png_sha256']:raise ValueError('shared source preview differs')
        else:shutil.copyfile(s['png'],source)
        with Image.open(source) as im:
            im.load()
            if im.size!=(w,h) or im.convert('RGB').tobytes()!=raw:raise ValueError('independent source readback differs')
        panes=[dict(label='Source',file=source.name,sha256=sha(source))]
        for alias in ['A','B']:
            pair=request['pairs'][keymap[cid]['models'][alias]]
            for m,mode in enumerate(['minimal','maximal']):
                stem=a.output/f'{cid}-{alias}-{mode}';retry=0
                while stem.with_suffix('.png').exists() or stem.with_suffix('.rgb').exists():
                    retry+=1;stem=a.output/f'{cid}-{alias}-{mode}-retry{retry}'
                result=render(raw,w,h,w*3,s['native_size'],pair[m],request['ratio'],stem,a.resizer,a.encoder,unit=512//max(map(int,request['ratio'].split(':'))));panes.append(dict(label=alias+' '+mode,file=Path(result['path']).name,sha256=result['sha256'],rgb_sha256=result['rgb_sha256'],width=result['width'],height=result['height']))
        case=dict(case_id=cid,id=s['id'],ratio=request['ratio'],source=s['source'],panes=panes);cases.append(case)
        completed_path.write_text(json.dumps(dict(cases=cases,blind_key_sha256=sha(keypath),reviewer='Codex coordinator; not human or independent implementer',api_calls=0,test_inference=False),indent=2)+'\n')
    for case in cases:
        source=case['source'];link=source['OriginalLandingURL'];figures=''.join(f'<figure><img src="{p["file"]}"><figcaption>{p["label"]}</figcaption></figure>' for p in case['panes']);articles.append(f'<article><h2>{case["case_id"]} {case["ratio"]}</h2><a href="{html.escape(link,quote=True)}">Original source</a><div>{figures}</div></article>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Blinded people crop review</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Blinded people crop review</h1><p>Editorial purpose. Compare each complete nested pair for composition, context, coherent subject and introduced face cuts. Other people can be excluded entirely. A/B identities remain hidden until every observation is sealed. Source-edge truncation is distinct from introduced crop cuts.</p>'+''.join(articles))
    total=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    if total>64*1024**2:raise ValueError('64MiB blind pixel budget exceeded')
    print(json.dumps(dict(cases=32,source_previews=16,crop_pngs=128,pngs=144,asset_bytes=total,blind_key_sha256=sha(keypath))))


if __name__=='__main__':main()
