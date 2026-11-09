#!/usr/bin/env python3
"""Bounded changed-crop witnesses with actual owned pixels and source links."""
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
    for name in ['cache','control','native','links','resizer','encoder','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3' or sha(a.control/'SHA256SUMS.json')!='4178e370857f0fe12f994de3eb5b6c10233dad690b983bd2465bdae24b1dd262':raise ValueError('fixed control/cache differs')
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned executable differs')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(cache['sources'])};records={}
    for arm in ['pose','pose-relations','pose-permuted-relations']:
        state=torch.load(a.control/'fits'/(arm+'-seed31')/'selected.pt',map_location='cpu',weights_only=False)
        records[arm]={(r['id'],r['query']):r for r in state['rows'] if r['essential']}
    if any(set(v)!=set(records['pose']) for v in records.values()):raise ValueError('matched essential query keys required')
    candidates=[]
    for key,real in records['pose-relations'].items():
        other=records['pose-permuted-relations'][key]
        if real['selected']==other['selected']:continue
        i=lookup[key[0]];q=key[1];boxes=cache['query_boxes'][i,q]
        movement=float((boxes[real['selected']]-boxes[other['selected']]).abs().max()*128)
        category='improvement' if other['cut'] and not real['cut'] else 'regression' if real['cut'] and not other['cut'] else 'changed-without-cut-label-change'
        candidates.append(dict(id=key[0],query=q,category=category,max_edge_movement_128_grid=movement))
    selected=[];used=set();available={}
    for category,limit in [('improvement',2),('regression',2),('changed-without-cut-label-change',1)]:
        pool=sorted([r for r in candidates if r['category']==category],key=lambda r:(-r['max_edge_movement_128_grid'],r['id'],r['query']));available[category]=len(pool);count=0
        for row in pool:
            if row['id'] in used:continue
            selected.append(row);used.add(row['id']);count+=1
            if count==limit:break
    native={r['source_id']:r for r in json.loads((a.native/'manifest.json').read_text())['images']};links=json.loads(a.links.read_text())
    a.output.mkdir(parents=True,exist_ok=False);articles=[];rows=[]
    escape=lambda value:html.escape(str(value),quote=True)
    for ordinal,row in enumerate(selected):
        sid,q=row['id'],row['query'];i=lookup[sid];source=native[sid];directory=a.output/('case'+str(ordinal));directory.mkdir()
        if source['family_id']!=cache['sources'][i]['family_id'] or source['role']!='dev':raise ValueError('DEV family identity differs')
        raw=Path(source['rgb']).read_bytes()
        if sha(Path(source['rgb']))!=source['rgb_sha256']:raise ValueError('owned RGB differs')
        original=Path(source['display']);shutil.copy2(original,directory/'source.png')
        with Image.open(original) as image:
            image.load()
            if image.size!=(source['width'],source['height']) or image.convert('RGB').tobytes()!=raw:raise ValueError('source PNG/RGB mismatch')
        ratio=cache['ratios'][q//2];scale=torch.tensor([*source['native_size'],*source['native_size']]);images=[]
        for arm in ['pose','pose-relations','pose-permuted-relations','contain']:
            record=records[arm][(sid,q)] if arm!='contain' else None
            box=(cache['query_boxes'][i,q,record['selected']]*scale).round().int().tolist() if record else None
            p,qratio=map(int,ratio.split(':'));unit=min(192,768//max(p,qratio))
            rendered=render(raw,source['width'],source['height'],source['width']*3,source['native_size'],box,ratio,directory/arm,a.resizer,a.encoder,unit=unit)
            images.append(dict(arm=arm,native_box=box,weak_record=record,**rendered))
        info=links[sid];metadata=dict(row,role='dev',family_id=source['family_id'],ratio=ratio,mode='maximal' if q%2 else 'minimal',source_information=info,images=images)
        rows.append(metadata);original_info=info['source']
        figures=''.join('<figure><img src="'+escape(str(Path(im['path']).relative_to(a.output)))+'"><figcaption>'+escape(im['arm'])+'<br>'+escape(im['weak_record'])+'</figcaption></figure>' for im in images)
        articles.append('<article id="case'+str(ordinal)+'"><h2>'+escape(sid)+' '+ratio+' '+metadata['mode']+' · '+row['category']+'</h2><p><a href="'+escape(original_info['OriginalLandingURL'])+'">'+escape(original_info['Title'])+'</a> · '+escape(original_info['Author'])+' · <a href="'+escape(original_info['License'])+'">license metadata</a></p><img class="source" src="case'+str(ordinal)+'/source.png"><div class="grid">'+figures+'</div></article>')
    total=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    if total>64*1024**2:raise ValueError('64MiB witness budget exceeded')
    document='<!doctype html><meta charset="utf-8"><title>Relationship crop witnesses</title><style>body{background:#20232a;color:#eee;font:16px system-ui;margin:24px}a{color:#9dd6ff}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;height:300px;object-fit:contain}.source{max-width:700px;height:440px}article{border-top:1px solid #999;padding:20px 0}figcaption{overflow-wrap:anywhere;font-size:13px}</style><h1>Relationship crop witnesses</h1><p>Selected changed DEV outputs, seed31; two weak-label improvements, up to two regressions and one other change, largest edge movement first with distinct source families. This is a diagnostic selection, not a quality sample. Head-cut labels are incomplete weak annotations. Actual pixels use owned cropping/resizing/PNG encoding and independent exact readback. No model promoted.</p>'+''.join(articles)
    (a.output/'index.html').write_text(document)
    result=dict(witnesses=rows,available_changed_queries=available,asset_bytes=total,budget_bytes=64*1024**2,control_sha256=sha(a.control/'SHA256SUMS.json'),api_calls=0,test_inference=False,model_promoted=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(witnesses=len(rows),available=available,asset_bytes=total)))


if __name__=='__main__':main()
