#!/usr/bin/env python3
"""Bounded soft-risk repair, introduced cut and source applicability witnesses."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
import torch
from PIL import Image
from artifact_identity import sha
from person_risk_inputs import load
from candidate_cut_head import CutHead,features
from evaluate_person_risk import choose
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','study','miap_objects','unsplash_objects','resizer','encoder']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned pixel binaries differ')
    contexts,fits,people,identities=load(a.fits,a.miap_objects,a.unsplash_objects);selections=json.loads((a.study/'selected.json').read_text());selected={r['model']:r for r in selections if r['applicability']=='person'};names=['hidden-attention-seed137','permuted-attention-seed137'];states={}
    for name in names:
        entry=next(r for r in fits if f"{r['arm']}-seed{r['seed']}"==name);state=torch.load(a.fits/name/'selected.pt',map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);states[name]={key:model(features(ctx['data'],entry['arm'])).reshape(len(ctx['data']['sources']),11,2,128) for key,ctx in contexts.items()}
    choices={}
    for cohort,ctx in contexts.items():
        logits=states[names[0]][cohort];strength=selected[names[0]]['strength'];choices[cohort]={'unfiltered':choose(ctx,logits,'all',0.)[0],names[0]:choose(ctx,logits,'person',strength)[0],'matched-risk-everywhere':choose(ctx,logits,'all',strength)[0],names[1]:choose(ctx,states[names[1]][cohort],'person',selected[names[1]]['strength'])[0]}
    new=contexts['new'];c=new['data'];base=choices['new']['unfiltered'];actual=choices['new'][names[0]];events=[]
    for i,source in enumerate(c['sources']):
        if source['role']!='dev':continue
        for q in range(18):
            if not new['metadata']['head_known'][i,q] or not new['metadata']['reference_known'][i,q]:continue
            ri,mode=divmod(q,2);x=int(base[i,ri,mode]);y=int(actual[i,ri,mode]);before=bool(c['negative'][i,q,x]);after=bool(c['negative'][i,q,y]);movement=float(((c['query_boxes'][i,q,x]-c['query_boxes'][i,q,y]).abs()*128).max());events.append(dict(cohort='new',id=source['id'],index=i,ratio_index=ri,query=q,before=before,after=after,movement=movement))
    repair=min([r for r in events if r['before'] and not r['after']],key=lambda r:(-r['movement'],r['id'],r['query']));introduced=min([r for r in events if not r['before'] and r['after']],key=lambda r:(r['ratio_index'] not in [7,8],-r['movement'],r['id'],r['query']));old=contexts['old'];oc=old['data'];old_events=[]
    for i,source in enumerate(oc['sources']):
        if old['person'][i]:continue
        for q in range(18):
            if not oc['query_valid'][i,q]:continue
            ri,m=divmod(q,2);x=int(choices['old']['matched-risk-everywhere'][i,ri,m]);y=int(choices['old'][names[0]][i,ri,m]);old_events.append(dict(cohort='old',id=source['id'],index=i,ratio_index=ri,query=q,target_score_recovery=float(oc['targets'][i,q,y]-oc['targets'][i,q,x])))
    recovered=min(old_events,key=lambda r:(-r['target_score_recovery'],r['id'],r['query']));requests=[dict(repair,reason='largest accepted weak-head repair'),dict(introduced,reason='introduced weak head cut; prioritize3:1/1:3'),dict(recovered,reason='largest original no-person target-score recovery through applicability')];output=a.study/'witnesses';output.mkdir(exist_ok=False);cases=[];articles=[]
    for number,request in enumerate(requests):
        ctx=contexts[request['cohort']];source=ctx['data']['sources'][request['index']];w,h=source['working_size'];rawpath=Path(source.get('rgb',str(Path(source['png']).with_suffix('.rgb'))));pixels=rawpath.read_bytes()
        if sha(rawpath)!=source['rgb_sha256']:raise ValueError('source pixels differ')
        png=output/f'case{number}-source.png';shutil.copy2(source['png'],png)
        with Image.open(png) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=pixels:raise ValueError('source PNG readback differs')
        panes=[dict(name='source',path=str(png),sha256=sha(png))];figures=[f'<figure><img src="{png.name}"><figcaption>Full source</figcaption></figure>'];ri=request['ratio_index'];ratio=ctx['data']['ratios'][ri];unit=384//max(map(int,ratio.split(':')));policies=['unfiltered',names[0],names[1]] if request['cohort']=='new' else ['unfiltered','matched-risk-everywhere',names[0]]
        for policy in policies:
            pair=[]
            for mode in range(2):
                candidate=int(choices[request['cohort']][policy][request['index'],ri,mode]);native=ctx['native'][request['index'],ri,mode,candidate].tolist();pair.append(native);p,r=map(int,ratio.split(':'))
                if (native[2]-native[0])*r!=(native[3]-native[1])*p:raise ValueError('exact aspect differs')
                name=policy+('-minimal' if mode==0 else '-maximal');result=render(pixels,w,h,w*3,source['native_size'],native,ratio,output/f'case{number}-{name}',a.resizer,a.encoder,unit=unit);panes.append(dict(name=name,native_box=native,candidate=candidate,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{name}</figcaption></figure>')
            outer,inner=pair
            if not (outer[0]<=inner[0]<inner[2]<=outer[2] and outer[1]<=inner[1]<inner[3]<=outer[3] and (inner[2]-inner[0])*(inner[3]-inner[1])<(outer[2]-outer[0])*(outer[3]-outer[1])):raise ValueError('strict nesting differs')
        cases.append(dict(**request,ratio=ratio,panes=panes,source=source['source']));url=html.escape(source['source']['source_url'],quote=True);articles.append(f'<article><h2>{html.escape(request["id"])} {ratio}</h2><p>{request["reason"]}</p><a href="{url}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size>16*1024*1024:raise ValueError('bounded witness storage exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=cases,asset_bytes=size,api_calls=0,test_inference=False,training_eligible=False,model_promoted=False),indent=2)+'\n');(output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Person-conditional risk witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Person-conditional soft risk</h1><p>Three selected diagnostic requests. A repaired weak head flag does not certify complete faces or artistic quality. Remaining failures are retained.</p>'+''.join(articles));print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),distinct_pngs=len({p['sha256'] for c in cases for p in c['panes']}),asset_bytes=size)))


if __name__=='__main__':main()
