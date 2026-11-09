#!/usr/bin/env python3
"""Bounded FIT-only matching-aspect nested crop options, without quality labels."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import torch
from PIL import Image
from artifact_identity import sha
from distill_attention_linear import Student
from subject_policy import face_relation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import pixels
from rendered_pixels import render


def nested(inner,outer):
    return outer[0]<=inner[0] and outer[1]<=inner[1] and outer[2]>=inner[2] and outer[3]>=inner[3]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','direct','control','resizer','encoder','output']:p.add_argument(name,type=Path)
    p.add_argument('--tighter-option',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3' or sha(a.direct)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':raise ValueError('fixed FIT caches differ')
    if sha(a.control/'SHA256SUMS.json')!='941d64dca4c3f0cf5477fdd78ac332dee1a080818c825e44b7c8f17043bb6fc4':raise ValueError('fixed rehearsal packet differs')
    weights=a.control/'relationship-rehearsal/replay-contrastive-seed31/selected.pt'
    if sha(weights)!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('registered weights/pixel executables differ')
    c=torch.load(a.cache,map_location='cpu',weights_only=False);d=torch.load(a.direct,map_location='cpu',weights_only=False)
    if c['sources']!=d['sources']:raise ValueError('source identities or roles differ')
    eligible=[i for i,s in enumerate(c['sources']) if s['role']=='fit' and s['annotation_valid'] and s['training_eligible']]
    selected=[];families=set()
    for count,needed in [(1,2),(2,1),(3,1)]:
        pool=sorted([i for i in eligible if min(len(d['head_rois'][i]),3)==count],key=lambda i:hashlib.sha256(('fit-pair-review-v1|'+c['sources'][i]['id']).encode()).hexdigest())
        found=0
        for i in pool:
            if c['sources'][i]['family_id'] in families:continue
            selected.append(i);families.add(c['sources'][i]['family_id']);found+=1
            if found==needed:break
        if found!=needed:raise ValueError('registered independent FIT strata unavailable')
    state=torch.load(weights,map_location='cpu',weights_only=False)['model'];model=Student(c['scorer_state'],state['teacher'],state['mean'],state['std'],nonlinear=True);model.load_state_dict(state);model.eval()
    scores=model(c['hidden'][selected],None)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'selection.json').write_text(json.dumps(dict(sources=[c['sources'][i] for i in selected],cache_sha256=sha(a.cache),direct_sha256=sha(a.direct),weights_sha256=sha(weights),api_calls=0,test_inference=False,training_labels_admitted=False),indent=2)+'\n')
    cases=[];articles=[]
    for local,i in enumerate(selected):
        source=c['sources'][i];w,h=source['native_size'];raw=Path(source['rgb'] if 'rgb' in source else str(Path(source['png']).with_suffix('.rgb'))).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['rgb_sha256']:raise ValueError('FIT source pixels changed')
        sw,sh=source['working_size'];source_png=a.output/f'source{local}.png';shutil.copy2(source['png'],source_png)
        with Image.open(source_png) as image:
            image.load()
            if image.size!=(sw,sh) or image.convert('RGB').tobytes()!=raw:raise ValueError('source PNG readback differs')
        heads=[pixels(r,w,h,outward=True) for r in d['head_rois'][i]]
        for ratio in ['3:1','1:3']:
            q=c['ratios'].index(ratio)*2;p,r=map(int,ratio.split(':'));unit=512//max(p,r);scale=torch.tensor([w,h,w,h]);outer=(c['query_boxes'][i,q]*scale).round().int().tolist();inner=(c['query_boxes'][i,q+1]*scale).round().int().tolist()
            safe=[]
            for query,boxes in [(q,outer),(q+1,inner)]:
                relation=[[face_relation(b,head) for head in heads] for b in boxes]
                cut=torch.tensor(['cut' in rel for rel in relation])&c['eligible'][i,query]
                if not torch.equal(cut,c['negative'][i,query]):raise ValueError('weak native head geometry differs')
                safe.append([k for k,b in enumerate(boxes) if c['eligible'][i,query,k] and 'cut' not in relation[k] and 'contained' in relation[k] and (b[2]-b[0])*r==(b[3]-b[1])*p])
            maximum=min(w//p,h//r);pairs=[]
            for oi in safe[0]:
                ob=outer[oi]
                if ob[2]-ob[0]!=p*maximum or ob[3]-ob[1]!=r*maximum:continue
                for ii in safe[1]:
                    ib=inner[ii];area=(ib[2]-ib[0])*(ib[3]-ib[1])
                    if not nested(ib,ob) or area>=(ob[2]-ob[0])*(ob[3]-ob[1]):continue
                    pose=c['extras'][i,q,oi,:16]+c['extras'][i,q+1,ii,:16]
                    pairs.append(dict(outer=oi,inner=ii,score=float(scores[local,q,oi]+scores[local,q+1,ii]),pose=[float(pose[k]) for k in [9,3,2]],inner_area=area))
            options=[];used=set()
            for strategy in ['student','pose','context']:
                ordered=sorted(pairs,key=lambda x:((-x['score'],) if strategy=='student' else tuple(-v for v in x['pose'])+(-x['score'],) if strategy=='pose' else (-x['inner_area'],-x['score']))+(x['outer'],x['inner']))
                for pair in ordered:
                    identity=tuple(outer[pair['outer']]+inner[pair['inner']])
                    if identity in used:continue
                    used.add(identity);options.append(dict(strategy=strategy,**pair));break
            if a.tighter_option:
                tighter=[pair for pair in pairs if .36<=pair['inner_area']/((outer[pair['outer']][2]-outer[pair['outer']][0])*(outer[pair['outer']][3]-outer[pair['outer']][1]))<=.64]
                for pair in sorted(tighter,key=lambda x:(-x['score'],x['outer'],x['inner'])):
                    identity=tuple(outer[pair['outer']]+inner[pair['inner']])
                    if identity in used:continue
                    options.append(dict(strategy='tighter',**pair));break
            directory=a.output/f'case{len(cases)}';directory.mkdir();panes=[]
            def draw(name,box,record):
                asset=render(raw,sw,sh,sw*3,[w,h],box,ratio,directory/name,a.resizer,a.encoder,unit=unit)
                pane=dict(name=name,native_box=box,record=record,**asset);panes.append(pane);return pane
            raw_indices=[int(scores[local,query].masked_fill(~c['eligible'][i,query],-torch.inf).argmax()) for query in [q,q+1]]
            for mode,k,boxes in zip(['minimal','maximal'],raw_indices,[outer,inner]):draw('raw-'+mode,boxes[k],dict(candidate=k))
            for option_index,pair in enumerate(options):
                pair['images']=[draw(f'option{option_index}-outer',outer[pair['outer']],dict(candidate=pair['outer'])),draw(f'option{option_index}-inner',inner[pair['inner']],dict(candidate=pair['inner']))]
            case=dict(id=source['id'],family_id=source['family_id'],role='fit',ratio=ratio,query=q,source=source['source'],source_png=str(source_png),source_png_sha256=sha(source_png),weak_head_nominations=len(heads),available_nested_pairs=len(pairs),options=options,panes=panes,raw_indices=raw_indices,raw_nested=nested(inner[raw_indices[1]],outer[raw_indices[0]]),training_eligible=False)
            cases.append(case)
            figures=''.join(f'<figure><img src="{Path(v["path"]).relative_to(a.output)}"><figcaption>{html.escape(v["name"])}</figcaption></figure>' for v in panes)
            articles.append(f'<article><h2>{html.escape(source["id"])} {ratio}</h2><a href="{html.escape(source["source"]["source_url"],quote=True)}">Original source</a><img class="source" src="{source_png.name}"><p>{len(pairs)} weak-geometry-compatible nested pairs. Raw pair nested: {case["raw_nested"]}. No artistic labels admitted.</p><div>{figures}</div></article>')
    size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    if size>128*1024*1024:raise ValueError('bounded FIT review storage exceeded')
    (a.output/'summary.json').write_text(json.dumps(dict(cases=cases,tighter_option=a.tighter_option,asset_bytes=size,budget_bytes=128*1024*1024,api_calls=0,test_inference=False,training_labels_admitted=False,model_promoted=False),indent=2)+'\n')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FIT nested-pair review</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}.source{max-width:768px;max-height:480px}article div{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}figure img{width:100%;height:300px;object-fit:contain}</style><h1>FIT-only nested-pair review</h1><p>Four fixed FIT families. Each curated option has matching-aspect outer Minimal and strictly smaller nested inner Maximal. Raw choices are separate diagnostics. Weak head geometry is incomplete. No labels, excellent intervals or evaluation-role changes.</p>'+''.join(articles))
    print(json.dumps(dict(sources=4,cases=len(cases),paired_options=sum(len(c['options']) for c in cases),asset_bytes=size)))


if __name__=='__main__':main()
