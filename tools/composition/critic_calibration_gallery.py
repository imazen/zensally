#!/usr/bin/env python3
"""Owned-pixel source context for all six frozen local-critic calibration pairs."""
import argparse
import html
import json
from pathlib import Path
import subprocess
from PIL import Image
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','admitted','resizer','encoder','output']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if sha(a.plan)!='51b788924c962a93ac9e8ec069ec72a147bf08baa06e7fdb5342e2153cc5b901' or sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('frozen plan/pixel tools differ')
    plan=json.loads(a.plan.read_text());sources=json.loads(a.admitted.read_text())['sources'];groups={}
    for row in plan['images']:
        if row['kind']=='candidate':groups.setdefault(row['task_id'],[]).append(row)
    if len(groups)!=6:raise ValueError('all six fixed calibration pairs required')
    a.output.mkdir(parents=True,exist_ok=False);cases=[];articles=[]
    def encode(rawpath,w,h,path):
        raw=rawpath.read_bytes()
        if len(raw)!=w*h*3:raise ValueError('bounded RGB length differs')
        subprocess.run([str(a.encoder),'--from-rgb',str(w),str(h),str(rawpath),str(path)],check=True,stdout=subprocess.PIPE)
        with Image.open(path) as image:
            image.load()
            if image.size!=(w,h) or image.convert('RGB').tobytes()!=raw:raise ValueError('independent PNG readback differs')
        return dict(path=str(path),sha256=sha(path),width=w,height=h,rgb_sha256=sha(rawpath))
    for number,(task,rows) in enumerate(groups.items()):
        if len(rows)!=2 or {r['weak_teacher_acceptable'] for r in rows}!={True,False}:raise ValueError('one positive/negative per fixed pair required')
        source=next(s for s in sources if any(c['task_id']==task for c in s['candidates']));rawpath=Path(source['tensor_path']).parent/'source.rgb';raw=rawpath.read_bytes();w,h=source['width'],source['height']
        if len(raw)!=w*h*3 or any(sha(rawpath)!=r['source_rgb_sha256'] for r in rows):raise ValueError('whole source identity differs')
        factor=384/max(w,h);dw,dh=max(1,round(w*factor)),max(1,round(h*factor));resized=subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),f'0,0,{w},{h},{dw},{dh}'],input=raw);preview=a.output/f'case{number}-source.rgb';preview.write_bytes(resized);panes=[dict(name='whole source, bounded context preview',**encode(preview,dw,dh,a.output/f'case{number}-source.png'))]
        for row in sorted(rows,key=lambda r:not r['weak_teacher_acceptable']):
            pixel=Path(row['rgb'])
            if sha(pixel)!=row['sha256']:raise ValueError('critic crop pixels differ')
            label='cached-accepted' if row['weak_teacher_acceptable'] else 'cached-rejected';panes.append(dict(name=label,id=row['id'],xywh=row['xywh'],**encode(pixel,row['width'],row['height'],a.output/f'case{number}-{label}.png')))
        cases.append(dict(task_id=task,source={k:v for k,v in source.items() if k!='candidates'},panes=panes));figures=''.join(f'<figure><img src="{Path(v["path"]).name}"><figcaption>{html.escape(v["name"])}</figcaption></figure>' for v in panes);articles.append(f'<article><h2>{html.escape(task)}</h2><div>{figures}</div></article>')
    size=sum(p.stat().st_size for p in a.output.iterdir() if p.is_file())
    if size>8*1024*1024:raise ValueError('eight-MiB witness allowance exceeded')
    (a.output/'summary.json').write_text(json.dumps(dict(cases=cases,plan_sha256=sha(a.plan),admitted_sha256=sha(a.admitted),asset_bytes=size,labels='existing unreviewed calibration judgments; no new label admission',api_calls=0,test_inference=False,model_promoted=False),indent=2)+'\n');(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Critic calibration pixels</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}article div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:448px;object-fit:contain}</style><h1>Six fixed critic calibration pairs</h1><p>Source context beside the exact owned RGB crops supplied to all critic variants. Cached accept/reject judgments are weak and unreviewed; no human or excellent-quality claim follows.</p>'+''.join(articles));print(json.dumps(dict(cases=len(cases),panes=sum(len(c['panes']) for c in cases),asset_bytes=size)))


if __name__=='__main__':main()
