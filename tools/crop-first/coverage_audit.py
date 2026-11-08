#!/usr/bin/env python3
"""Companion candidate-coverage audit; main labels and geometry stay immutable."""
import argparse,csv,json,subprocess
from pathlib import Path
from types import SimpleNamespace
from expanded import load,save,render


def broaden(pool,width,height,mode,ratio):
    result=[{**c,'generators':list(c['generators'])} for c in pool]
    seen={(c['x'],c['y'],c['w'],c['h']) for c in result};serial=0
    rw,rh=ratio;k=min(width//rw,height//rh);horizontal=width*rh>height*rw
    for scale,positions in ([(1,[0,1])] if mode=='minimal' else [(1,[.2,.35,.65,.8]),(.75,[.25,.75])]):
        unit=max(1,int(k*scale));w,h=rw*unit,rh*unit
        for p in positions:
            x=round((width-w)*(p if horizontal else .5));y=round((height-h)*(.5 if horizontal else p));box=x,y,w,h
            if box in seen:continue
            seen.add(box);serial+=1
            result.append({'id':f'A{serial:02}','x':x,'y':y,'w':w,'h':h,'generators':['coverage-axis-control'],'reason':'Additional spatial coverage at unchanged crop size'})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path);p.add_argument('root',type=Path);p.add_argument('executable',type=Path);a=p.parse_args()
    sources=load(a.inputs/'sources.json');selected=[next(s for s in sources if s['role']=='train' and s['content_group']==g) for g in ['1400','1600','2400','9226']]
    out=a.root/'coverage-audit';out.mkdir(exist_ok=False);(out/'batches').mkdir()
    indexed={s['id']:s for p in sorted((a.root/'batches').glob('??-sources.json')) for s in load(p)}
    direct={r['task_id']:r for p in sorted((a.root/'batches').glob('??-proposals.json')) for r in load(p)}
    save(out/'protocol.json',{'source_ids':[s['id'] for s in selected],'roles_unchanged':True,'ratios':[[1,8],[8,1]],'modes':['minimal','maximal'],'required_tasks':16,'change':'all existing interior/scale proposals plus primary-axis positions at unchanged crop sizes','face_criteria_changed':False,'main_run':'../','admission':'unreviewed','training_eligible':False,'face_safety_certified':False})
    save(out/'sources.json',selected)
    for i in range(0,4,2):
        batch=[];proposals=[]
        for source in selected[i:i+2]:
            original=indexed[source['id']];tasks=[t for t in original['tasks'] if t['target_ratio'] in [[1,8],[8,1]]]
            batch.append({**original,'tasks':tasks});proposals.extend(direct[t['task_id']] for t in tasks)
        name=f'{i//2:02}';path=out/'batches'/f'{name}-sources.json';save(path,batch);prop=out/'batches'/f'{name}-proposals.json';save(prop,proposals)
        render(SimpleNamespace(inputs=a.inputs,output=out,batch=path,proposals=prop,executable=a.executable,full_pool=True))
    print(json.dumps({'coverage_audit_sources':4,'coverage_audit_tasks':16}))


if __name__=='__main__':main()
