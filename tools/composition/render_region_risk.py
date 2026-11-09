#!/usr/bin/env python3
"""Fixed risk-head witness: a new cut and the largest original quality-proxy loss."""
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
from person_risk_inputs import load
from region_risk_features import build,features,head_applicability
from candidate_cut_head import CutHead
from evaluate_person_risk import choose
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fits','routing','parent','miap','unsplash','resizer','encoder','output']:p.add_argument(name,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('pixel tools differ')
    contexts,_,_,_=load(a.parent,a.miap,a.unsplash);name='hidden-regions-attention-seed137';chosen=next(r for r in json.loads((a.routing/'selected.json').read_text()) if r['model']==name);proposal=json.loads((a.fits/'proposals.json').read_text());entry=next(r for r in json.loads((a.fits/'summary.json').read_text())['fits'] if r['arm']=='hidden-regions-attention' and r['seed']==137);path=a.fits/name/'selected.pt'
    if sha(path)!=entry['checkpoint_sha256']:raise ValueError('fixed diagnostic model changed')
    if json.loads((a.routing/'protocol.json').read_text()).get('measured_head_required',False):head_applicability(contexts,proposal['cohorts'])
    state=torch.load(path,map_location='cpu',weights_only=False)['model'];model=CutHead(state['mean'],state['std']);model.load_state_dict(state);lookups={};sources={}
    for cohort,ctx in contexts.items():
        c=ctx['data'];real=build(ctx,proposal['cohorts'][cohort]);permuted=build(ctx,proposal['cohorts'][cohort],permuted=True);x=features(c,real,permuted,'hidden-regions-attention');logits=model(x).reshape(len(c['sources']),11,2,128);after,present=choose(ctx,logits,'person',chosen['strength']);before,full=choose(ctx,torch.full_like(logits,-1.),'all',0.)
        if not present.all() or not full.all():raise ValueError('soft witness must preserve offers')
        lookups[cohort]=(before,after);sources[cohort]={s['id']:(i,s) for i,s in enumerate(c['sources'])}
    prediction=json.loads((a.routing/'predictions.json').read_text());newrows=[r for r in prediction if r['model']==name and r['query']<18 and r['head_known'] and r['reference_known'] and r['weak_cut'] and not r['parent_cut']]
    if not newrows:raise ValueError('registered new-cut witness missing')
    def movement(row):
        i,s=sources['new'][row['id']];q=row['query'];before=lookups['new'][0];native=contexts['new']['native'];ri=q//2;m=q%2;return max(abs(x-y)*128/z for x,y,z in zip(row['native'],native[i,ri,m,int(before[i,ri,m])].tolist(),[*s['native_size'],*s['native_size']]))
    bad=max(newrows,key=lambda r:(movement(r),r['id'],r['query']));ctx=contexts['old'];before,after=lookups['old'];c=ctx['data'];regressions=[]
    for i,s in enumerate(c['sources']):
        for q in range(18):
            if not c['query_valid'][i,q]:continue
            ri,m=divmod(q,2);a0=int(before[i,ri,m]);a1=int(after[i,ri,m]);regressions.append((float(c['targets'][i,q,a0]-c['targets'][i,q,a1]),s['id'],q))
    loss,sid,q=max(regressions,key=lambda r:(r[0],r[1],r[2]));requests=[('new','largest movement new nominated-head cut',bad['id'],bad['query']//2),('old','largest original target-score loss; not itself artistic ground truth',sid,q//2)]
    a.output.mkdir(parents=True,exist_ok=False);cases=[];articles=[]
    for index,(cohort,reason,sid,ri) in enumerate(requests):
        i,s=sources[cohort][sid];ctx=contexts[cohort];w,h=s['working_size'];rawpath=Path(s.get('rgb',str(Path(s['png']).with_suffix('.rgb'))));raw=rawpath.read_bytes()
        if sha(rawpath)!=s['rgb_sha256'] or sha(Path(s['png']))!=s['png_sha256']:raise ValueError('source pixels differ')
        source=a.output/f'case{index}-source.png';shutil.copyfile(s['png'],source)
        with Image.open(source) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=raw:raise ValueError('independent source readback differs')
        panes=[dict(name='source',path=str(source),sha256=sha(source))];figures=[f'<figure><img src="{source.name}"><figcaption>Whole source</figcaption></figure>'];ratio=ctx['data']['ratios'][ri]
        for arm,indices in [('parent',lookups[cohort][0]),('region-risk',lookups[cohort][1])]:
            for m in range(2):
                label=arm+('-minimal' if m==0 else '-maximal');box=ctx['native'][i,ri,m,int(indices[i,ri,m])].tolist();result=render(raw,w,h,w*3,s['native_size'],box,ratio,a.output/f'case{index}-{label}',a.resizer,a.encoder,unit=256//max(map(int,ratio.split(':'))));panes.append(dict(name=label,native_box=box,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{label}</figcaption></figure>')
        cases.append(dict(id=sid,cohort=cohort,ratio=ratio,reason=reason,source=s['source'],panes=panes));articles.append(f'<article><h2>{html.escape(sid)} {ratio}</h2><p>{reason}</p><a href="{html.escape(s["source"]["source_url"],quote=True)}">Original source</a><div>{"".join(figures)}</div></article>')
    write(a.output/'summary.json',dict(cases=cases,model=name,strength=chosen['strength'],largest_original_target_loss=loss,api_calls=0,test_inference=False,model_promoted=False));(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Learned region risk witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Region-risk suitability</h1><p>Fixed hidden-regions-attention seed137, FIT-selected strength. All offers retained; original gate outcomes are recorded separately. Original teacher scores are weak proxies, not human artistic labels.</p>'+''.join(articles))
    total=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    if total>8*1024**2:raise ValueError('bounded witness budget exceeded')
    print(json.dumps(dict(cases=[dict(id=c['id'],ratio=c['ratio']) for c in cases],panes=sum(len(c['panes']) for c in cases),bytes=total,largest_original_target_loss=loss)))


if __name__=='__main__':main()
