#!/usr/bin/env python3
"""Prepare portable cached API plans and actual rendered candidate evidence."""
import argparse,csv,json,math,os,subprocess
from pathlib import Path
from corpus import RATIOS,save,sha
from expanded import HEADER
from direct_zones import validate as validate_zones

def immutable(p,value):
    b=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=b:raise ValueError(f'immutable task changed: {p}')
    else:p.write_bytes(b)
def sources(root):return [json.loads(p.read_text()) for p in sorted(root.glob('*/source.json'))]
def image(p):return {'path':str(p),'sha256':sha(p.read_bytes())}

def zones(a):
    a.output.mkdir(parents=True,exist_ok=True);tasks=a.output/'tasks';tasks.mkdir(exist_ok=True);assets=a.output/'assets';assets.mkdir(exist_ok=True);jobs=[]
    selected=sources(a.inputs);selected=selected[:a.limit] if a.limit else selected
    for s in selected:
        source=Path(s['directory'])/'preview.png';dst=assets/(s['id']+'.png')
        if dst.exists() and sha(dst.read_bytes())!=s['preview_sha256']:raise ValueError('staged preview changed')
        if not dst.exists():dst.write_bytes(source.read_bytes())
        task={'id':s['id'],'role':s['bucket'],'use':s['use'],'family_id':s['family_id'],'width':s['width'],'height':s['height'],'ratios':RATIOS,'image':image(dst)}
        p=tasks/(s['id']+'.json');immutable(p,task);jobs.append({'id':'zones-'+s['id'],'kind':'zones','task':str(p),'model':'gpt-6.1-terra'})
    immutable(a.output/a.plan,jobs);print(json.dumps({'planned':len(jobs),'plan':str(a.output/a.plan)}))

def rectangles(s,zones):
    w,h=s['width'],s['height'];tasks=[]
    for t in zones['crops']:
        rw,rh=map(int,t['ratio'].split(':'));k0=min(w//rw,h//rh);seen=set();candidates=[]
        def add(cx,cy,k,origin):
            cw,ch=k*rw,k*rh;x=max(0,min(w-cw,round(cx-cw/2)));y=max(0,min(h-ch,round(cy-ch/2)));rect=(x,y,cw,ch)
            if rect in seen:return
            seen.add(rect);missed=[]
            for roi in zones['rois']:
                if roi['kind']=='face_head':
                    b=roi['box'];f=[math.floor(b['left']/100*w),math.floor(b['top']/100*h),math.ceil(b['right']/100*w),math.ceil(b['bottom']/100*h)]
                    if not(x<=f[0] and y<=f[1] and x+cw>=f[2] and y+ch>=f[3]):missed.append(roi['id'])
            candidates.append({'id':f'C{len(candidates):02}','xywh':list(rect),'origin':origin,'missed_proposed_faces':missed})
        for option in t['options'] if t['action']=='crop' else []:
            b=option['box'];k=k0 if t['mode']=='minimal' else max(math.ceil(k0*.5),min(k0,math.floor(min((b['right']-b['left'])/100*w/rw,(b['bottom']-b['top'])/100*h/rh))))
            add((b['left']+b['right'])/200*w,(b['top']+b['bottom'])/200*h,k,'terra-proposal')
        for anchor in [.5,1/3,2/3]:
            k=k0 if t['mode']=='minimal' else max(1,round(k0*.75))
            add(k*rw/2+(w-k*rw)*anchor,k*rh/2+(h-k*rh)*anchor,k,'interior-baseline')
        # Stable blind display order; teacher origin is never sent to reviewers.
        candidates.sort(key=lambda c:sha((s['id']+t['ratio']+t['mode']+str(c['xywh'])).encode()))
        for i,c in enumerate(candidates):c['id']=f'C{i:02}'
        tasks.append({'task_id':f'{s["id"]}-{rw}x{rh}-{t["mode"]}','ratio':t['ratio'],'mode':t['mode'],'proposed_action':t['action'],'candidates':candidates})
    return tasks

def render(a):
    a.output.mkdir(parents=True,exist_ok=True);jobs=[];sol_jobs=[]
    overrides=dict(s.split('=',1) for s in a.override_result)
    records=sources(a.inputs)
    if a.only:records=[s for s in records if s['id'] in a.only.split(',')]
    if not records:raise ValueError('explicit render selection is empty')
    for index,s in enumerate(records):
        result=Path(overrides[s['id']]) if s['id'] in overrides else a.results/('zones-'+s['id']+'.json')
        if not result.exists():raise ValueError(f'missing source proposal result: {s["id"]}')
        entry=Path(json.loads(result.read_text())['entry'])
        # Cache may be mirrored locally at a different root.
        if a.cache:entry=a.cache/entry.name
        zones=validate_zones(json.loads((entry/'response.json').read_text()),{'id':s['id'],'ratios':RATIOS});tasks=rectangles(s,zones);directory=a.output/s['id']
        if directory.exists():raise ValueError('fresh render output required')
        table=a.output/(s['id']+'.tsv')
        with table.open('x') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(HEADER)
            for t in tasks:
                rw,rh=map(int,t['ratio'].split(':'))
                for c in t['candidates']:
                    name=t['task_id']+'-'+c['id']+'.png';c['file']=name
                    writer.writerow([t['task_id'],s['id'],t['mode'],rw,rh,s['width'],s['height'],c['id'],*c['xywh'],name,c['origin']])
        subprocess.run([str(a.renderer),str(Path(s['directory'])/'source.png'),str(directory),s['id'],str(table),'768','65536'],check=True)
        for t in tasks:
            for c in t['candidates']:c['image']=image(directory/c['file'])
        original=image(directory/'source.png')
        packet={**s,'image':original,'rois':zones['rois'],'face_coverage':zones['face_coverage'],'tasks':tasks};save(directory/'packet.json',packet)
        def job(chunk,model,n):
            task={'id':s['id'],'role':s['bucket'],'use':s['use'],'family_id':s['family_id'],'width':s['width'],'height':s['height'],'image':original,'tasks':chunk}
            path=directory/(model+'-'+str(n)+'.json');immutable(path,task)
            return {'id':model+'-'+s['id']+'-'+str(n),'kind':'review','model':model,'task':str(path)}
        for n,start in enumerate(range(0,len(tasks),6)):jobs.append(job(tasks[start:start+6],'gpt-6.1-terra',n))
        # A deterministic TRAIN-only cross-model audit, before any fitting.
        if s['use']=='fit' and index%4==0:
            extreme=[t for t in tasks if t['ratio'] in ['3:1','1:3','6:1','1:6']]
            for n,start in enumerate(range(0,len(extreme),4)):sol_jobs.append(job(extreme[start:start+4],'gpt-6.1-sol',n))
        print(json.dumps({'rendered':s['id'],'tasks':len(tasks),'candidates':sum(len(t['candidates']) for t in tasks)}),flush=True)
    immutable(a.output/'terra-plan.json',jobs);immutable(a.output/'sol-plan.json',sol_jobs)
    print(json.dumps({'terra_requests':len(jobs),'sol_requests':len(sol_jobs)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['zones','render']);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('--plan',default='zones-plan.json');p.add_argument('--limit',type=int);p.add_argument('--only');p.add_argument('--override-result',action='append',default=[]);p.add_argument('--results',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--renderer',type=Path);a=p.parse_args()
    (zones if a.action=='zones' else render)(a)
if __name__=='__main__':main()
