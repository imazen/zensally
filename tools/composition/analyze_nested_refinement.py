#!/usr/bin/env python3
"""Paired availability diagnostics and bounded actual-pixel interval witnesses."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import sys
from PIL import Image
from artifact_identity import sha
from distill_attention_linear import write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render
from crop_corridors import samples,member


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['geometry','parent','resizer','encoder','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.parent/'SHA256SUMS.json')!='8e2456a64197e835ad58a366eb3fd8008ee11b0e629d4c18e7dfd9a4cff44ac2' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('frozen parent/pixel tool identity differs')
    rows=[json.loads(line) for line in (a.geometry/'predictions.jsonl').read_text().splitlines()];parents=[json.loads(line) for line in (a.geometry/'parents.jsonl').read_text().splitlines()];parents={(r['seed'],r['cohort'],r['id'],r['ratio_index']):r for r in parents};sources=json.loads((a.geometry/'sources.json').read_text());sources={(cohort,s['id']):s for cohort,items in sources.items() for s in items};before={}
    with (a.parent/'predictions.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['kind']=='soft' and r['applicability']=='person' and r['model'].startswith('hidden-attention-seed'):before[int(r['model'].split('seed')[1]),r['id'],r['query']]=r
    summaries=[];events=[]
    for seed in [31,137,911]:
        for policy in sorted({r['policy'] for r in rows}):
            for secondary in [False,True]:
                selected=[r for r in rows if r['seed']==seed and r['policy']==policy and r['cohort']=='new' and r['role']=='dev' and r['essential']!=secondary];counts=Counter()
                for r in selected:
                    for m in range(2):
                        if not r['head_known'] or not r['reference_known'][m]:continue
                        q=r['ratio_index']*2+m;old=before[seed,r['id'],q];offered=r['offered'][m];counts['requests']+=1;counts['parent_cuts']+=old['weak_cut'];counts['offered']+=offered;counts['cuts']+=r['weak_cut'][m];counts['matches']+=r['reference_match'][m]
                        if offered:
                            counts['common_parent_cuts']+=old['weak_cut'];counts['common_parent_matches']+=old['reference_match'];counts['repaired']+=old['weak_cut'] and not r['weak_cut'][m];counts['introduced']+=not old['weak_cut'] and r['weak_cut'][m]
                        else:counts['unavailable_parent_cuts']+=old['weak_cut']
                        if seed==137 and policy=='faces-pose/pad25' and not secondary:events.append(dict(row=r,mode=m,before_cut=old['weak_cut'],before_match=old['reference_match']))
                summaries.append(dict(seed=seed,policy=policy,secondary=secondary,**counts))
    def movement(e):return max((x or 0) for x in e['row']['movement_grid'])
    repair=min([e for e in events if e['before_cut'] and e['row']['offered'][e['mode']] and not e['row']['weak_cut'][e['mode']]],key=lambda e:(-movement(e),e['row']['id'],e['mode']))
    remaining=min([e for e in events if e['row']['weak_cut'][e['mode']]],key=lambda e:(e['row']['ratio'] not in ['3:1','1:3'],-movement(e),e['row']['id']))
    unavailable=min([e for e in events if not e['row']['offered'][e['mode']]],key=lambda e:(e['row']['ratio'] not in ['3:1','1:3'],e['row']['id'],e['mode']))
    a.output.mkdir(parents=True,exist_ok=False);gallery=a.output/'witnesses';gallery.mkdir();cases=[];articles=[]
    for number,(reason,event) in enumerate([('largest movement among offered head-cut repairs',repair),('remaining labelled cut; prioritize3:1/1:3',remaining),('accepted endpoint made unavailable; prioritize3:1/1:3',unavailable)]):
        r=event['row'];s=sources['new',r['id']];w,h=s['working_size'];rawpath=Path(s['png']).with_suffix('.rgb');pixels=rawpath.read_bytes()
        if sha(rawpath)!=s['rgb_sha256'] or sha(Path(s['png']))!=s['png_sha256']:raise ValueError('witness source changed')
        source_png=gallery/f'case{number}-source.png';shutil.copy2(s['png'],source_png)
        with Image.open(source_png) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=pixels:raise ValueError('source PNG independent readback differs')
        parent=parents[r['seed'],'new',r['id'],r['ratio_index']];panes=[dict(name='source',path=str(source_png),sha256=sha(source_png))];figures=[f'<figure><img src="{source_png.name}"><figcaption>Full source</figcaption></figure>'];requests=[('parent-minimal',parent['outer']),('parent-maximal',parent['inner']),('refined-minimal',r['outer']),('refined-maximal',r['inner'])]
        if all(r['offered']):
            middle=samples(r['inner'],r['outer'],r['ratio'])[2]
            if not member(middle,r['inner'],r['outer'],r['ratio']):raise ValueError('intermediate witness not in exact interval')
            requests.append(('refined-intermediate',middle))
        for name,box in requests:
            label=name if box is not None else name+' unavailable: contain preview only'
            result=render(pixels,w,h,w*3,s['native_size'],box,r['ratio'],gallery/f'case{number}-{name}',a.resizer,a.encoder,unit=384//max(map(int,r['ratio'].split(':'))));panes.append(dict(name=label,native_box=box,**result));figures.append(f'<figure><img src="{Path(result["path"]).name}"><figcaption>{label}</figcaption></figure>')
        cases.append(dict(id=r['id'],ratio=r['ratio'],reason=reason,record=r,source=s['source'],panes=panes));articles.append(f'<article><h2>{html.escape(r["id"])} {r["ratio"]}</h2><p>{reason}; {r["status"]}</p><a href="{html.escape(s["source"]["source_url"],quote=True)}">Original source</a><div>{"".join(figures)}</div></article>')
    size=sum(p.stat().st_size for p in gallery.iterdir() if p.is_file())
    if size>16*1024*1024:raise ValueError('witness storage budget exceeded')
    write(gallery/'summary.json',dict(cases=cases,asset_bytes=size,training_eligible=False,model_promoted=False));(gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Nested refinement witnesses</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}a{color:#9df}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:384px;object-fit:contain}</style><h1>Exact nested origin refinement</h1><p>Provided-head geometry is conditional on incomplete proposals. Intermediate frames are diagnostic samples, not excellent crop labels. Unavailable outputs show contain previews explicitly.</p>'+''.join(articles));write(a.output/'summary.json',dict(rows=summaries,api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(dict(primary=[r for r in summaries if r['policy']=='faces-pose/pad25'],cases=[dict(id=c['id'],ratio=c['ratio'],reason=c['reason']) for c in cases],asset_bytes=size)))


if __name__=='__main__':main()
