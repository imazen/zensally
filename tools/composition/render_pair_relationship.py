#!/usr/bin/env python3
"""Bounded actual pixels for pair correction gains, cuts and retained failures."""
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
    for name in ['fits','old','new','metadata','resizer','encoder']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    expected=[(a.old,'2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),(a.new,'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'),(a.resizer,'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041'),(a.encoder,'626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d')]
    if any(sha(path)!=digest for path,digest in expected):raise ValueError('frozen render inputs differ')
    old=torch.load(a.old,map_location='cpu',weights_only=False);new=torch.load(a.new,map_location='cpu',weights_only=False);data={'old':old,'new':new};meta=json.loads(a.metadata.read_text());baseline=torch.load(a.fits/'additive-linear-seed31/epoch00.pt',weights_only=False);states={'common-joint':baseline}
    for name in ['additive-linear-seed137','pair-real-seed137','pair-real-seed911']:states[name]=torch.load(a.fits/name/'selected.pt',weights_only=False)
    indices={cohort:[i for i,s in enumerate(c['sources']) if s['role']=='dev'] for cohort,c in data.items()};events=[]
    for name,state in states.items():
        for cohort,c in data.items():
            chosen=state['old_choices' if cohort=='old' else 'new_choices'];before=baseline['old_choices' if cohort=='old' else 'new_choices']
            for j,ri,mode in (chosen!=before).nonzero().tolist():
                i=indices[cohort][j];q=ri*2+mode;source=c['sources'][i];x=int(before[j,ri,mode]);y=int(chosen[j,ri,mode]);geometry=c['boxes'][i,ri] if cohort=='old' else c['query_boxes'][i,q];movement=float(((geometry[x]-geometry[y]).abs()*128).max());row=dict(model=name,cohort=cohort,id=source['id'],source_index=i,dev_index=j,ratio_index=ri,query=q,essential=q<18,before=x,after=y,edge_movement_grid=movement)
                if cohort=='new':row.update(valid=bool(c['reference_valid'][i,q].any()),before_cut=bool(c['negative'][i,q,x]),after_cut=bool(c['negative'][i,q,y]))
                else:row.update(valid=bool(c['query_valid'][i,q]),target_score_improvement=float(c['targets'][i,q,y]-c['targets'][i,q,x]))
                events.append(row)
    requests=[]
    for model,kind in [('pair-real-seed911','introduced'),('pair-real-seed137','repaired')]:
        candidates=[r for r in events if r['cohort']=='new' and r['model']==model and r['essential'] and r['valid'] and ((not r['before_cut'] and r['after_cut']) if kind=='introduced' else (r['before_cut'] and not r['after_cut']))]
        winner=min(candidates,key=lambda r:(-r['edge_movement_grid'],r['id'],r['query']));requests.append(dict(cohort='new',id=winner['id'],ratio_index=winner['ratio_index'],reason=kind+' weak-head cut; largest normalized edge movement'))
    candidates=[r for r in events if r['cohort']=='old' and r['model']=='pair-real-seed137' and r['essential'] and r['valid']];winner=min(candidates,key=lambda r:(-r['target_score_improvement'],r['id'],r['query']));requests.append(dict(cohort='old',id=winner['id'],ratio_index=winner['ratio_index'],reason='largest original essential endpoint target-score improvement'))
    requests.append(dict(cohort='new',id='openimages:9b15ccd240edae57',ratio_index=0,reason='fixed counter-group square failure from earlier review'))
    output=a.fits/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for number,request in enumerate(requests):
        cohort=request['cohort'];c=data[cohort];i=next(i for i,s in enumerate(c['sources']) if s['id']==request['id']);j=indices[cohort].index(i);source=next(s for s in meta[cohort] if s['id']==request['id']);w,h=source['working_size'];rawpath=Path(source.get('rgb',str(Path(source['png']).with_suffix('.rgb'))));pixels=rawpath.read_bytes()
        if sha(rawpath)!=source['rgb_sha256']:raise ValueError('source RGB differs')
        path=output/f'case{number}-source.png';shutil.copy2(source['png'],path)
        with Image.open(path) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=pixels:raise ValueError('source PNG readback differs')
        panes=[dict(name='source',path=str(path),sha256=sha(path))];figures=[f'<figure><img src="{path.name}"><figcaption>Full source</figcaption></figure>'];ri=request['ratio_index'];ratio=c['ratios'][ri];unit=384//max(map(int,ratio.split(':')));nw,nh=source['native_size']
        for model,state in states.items():
            choice=state['old_choices' if cohort=='old' else 'new_choices'][j,ri];pair=[]
            for mode in range(2):
                q=ri*2+mode;k=int(choice[mode]);box=c['boxes'][i,ri,k] if cohort=='old' else c['query_boxes'][i,q,k];native=(box*torch.tensor([nw,nh,nw,nh])).round().int().tolist();pair.append(native);p,r=map(int,ratio.split(':'))
                if (native[2]-native[0])*r!=(native[3]-native[1])*p:raise ValueError('native aspect differs')
                name=model+('-minimal' if mode==0 else '-maximal');result=render(pixels,w,h,w*3,source['native_size'],native,ratio,output/f'case{number}-{name}',a.resizer,a.encoder,unit=unit);panes.append(dict(name=name,native_box=native,candidate=k,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{name}</figcaption></figure>')
            outer,inner=pair
            if not (outer[0]<=inner[0]<inner[2]<=outer[2] and outer[1]<=inner[1]<inner[3]<=outer[3] and (inner[2]-inner[0])*(inner[3]-inner[1])<(outer[2]-outer[0])*(outer[3]-outer[1])):raise ValueError('strict nesting differs')
        cases.append(dict(**request,ratio=ratio,panes=panes,source=source['source']));url=html.escape(source['source']['source_url'],quote=True);articles.append(f'<article><h2>{html.escape(request["id"])} {ratio}</h2><p>{request["reason"]}</p><a href="{url}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>32*1024*1024:raise ValueError('bounded witness budget exceeded')
    (output/'events.json').write_text(json.dumps(events,indent=2)+'\n');(output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,api_calls=0,test_inference=False,training_eligible=False,model_promoted=False),indent=2)+'\n');(output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pair correction witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Pair correction witnesses</h1><p>Deliberately selected changes and one retained failure. Actual owned crop pixels; no artistic or complete-face guarantee.</p>'+''.join(articles));print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),distinct_pngs=len({p['sha256'] for c in cases for p in c['panes']}),asset_bytes=size)))


if __name__=='__main__':main()
