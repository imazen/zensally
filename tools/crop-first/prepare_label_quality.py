#!/usr/bin/env python3
"""Render a fixed balanced pilot using owned resampling and PNG encoding."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import torch
from PIL import Image
from subject_geometry import project
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'composition'))
from artifact_identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','study','attention','native','resizer','encoder']:p.add_argument(key,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':raise ValueError('owned executable differs')
    if sha(a.attention/'cache/cache.pt')!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':raise ValueError('fixed frozen C002 candidate cache differs')
    cache=torch.load(a.attention/'cache/cache.pt',map_location='cpu',weights_only=False);lookup={s['id']:i for i,s in enumerate(cache['sources'])}
    native={r['source_id']:r for r in json.loads((a.native/'manifest.json').read_text())['images']}
    paths=json.loads((a.study/'real-tasks/tasks.json').read_text());groups=defaultdict(list)
    for path in paths:
        task=json.loads(Path(path).read_text());groups[task['cell']].append(task)
    if len(groups)!=12 or len(lookup)!=72:raise ValueError('fixed twelve-cell72-source panel required')
    out=a.root/'rendered';out.mkdir(exist_ok=False);tasks=a.root/'tasks';tasks.mkdir();cases=[];requests=[];readbacks=[]
    for index,(cell,pool) in enumerate(sorted(groups.items())):
        role='fit' if index%2==0 else 'dev';source=min((s for s in pool if s['role']==role),key=lambda s:hashlib.sha256(('label-quality-v1|'+s['id']).encode()).hexdigest())
        sid=source['id'];image=native[sid];w,h=source['width'],source['height'];sw,sh=image['width'],image['height'];raw=Path(image['rgb']).read_bytes()
        if sha(Path(image['rgb']))!=image['rgb_sha256'] or len(raw)!=sw*sh*3 or sha(Path(source['image']['path']))!=source['image']['sha256']:raise ValueError('owned source changed')
        ratio='3:1' if index%4<2 else '1:3';mode='maximal' if index%2==0 else 'minimal';p,r=map(int,ratio.split(':'));q=cache['ratios'].index(ratio)*2+(mode=='maximal');ci=lookup[sid]
        if cache['sources'][ci]['role']!=role or cache['sources'][ci]['family_id']!=source['family_id']:raise ValueError('fixed role/family differs')
        specifications=[]
        for panel,folder in [('terra','real-terra'),('sol','sol-real72')]:
            record=json.loads((a.study/folder/(sid.replace(':','-')+'.json')).read_text());box=None;reason='invalid source annotation'
            if record['annotation_valid']:
                zones=json.loads((Path(record['entry'])/'zones.json').read_text());crop=next(c for c in zones['crops'] if c['ratio']==ratio and c['mode']==mode);reason=crop['action']
                if crop['action']!='contain' and crop['options']:box=project(crop['options'][0]['box'],w,h,ratio,mode)
            specifications.append(dict(origin=panel,native_box=box,fallback_reason=reason))
        selected=int(cache['initial_scores'][ci,q].masked_fill(~cache['eligible'][ci,q],-torch.inf).argmax());box=(cache['query_boxes'][ci,q,selected]*torch.tensor([w,h,w,h])).round().int().tolist()
        specifications.extend([dict(origin='frozen-C002',native_box=box,selected=selected),dict(origin='full-source-contain',native_box=None)])
        caseid=f'case{index:02}';directory=out/caseid;directory.mkdir();candidates=[]
        for spec in specifications:
            ow,oh=192*p,192*r
            if spec['native_box'] is None:
                scale=min(ow/sw,oh/sh);rw,rh=max(1,round(sw*scale)),max(1,round(sh*scale));bounds=[0,0,sw,sh]
            else:
                x0,y0,x1,y1=spec['native_box'];bounds=[max(0,math.floor(x0*sw/w)),max(0,math.floor(y0*sh/h)),min(sw,math.ceil(x1*sw/w)),min(sh,math.ceil(y1*sh/h))];rw,rh=ow,oh
            x0,y0,x1,y1=bounds;config=','.join(map(str,[x0,y0,x1-x0,y1-y0,rw,rh]))
            resized=subprocess.check_output([str(a.resizer),str(sw),str(sh),str(sw*3),config],input=raw)
            if len(resized)!=rw*rh*3:raise ValueError('owned crop byte length differs')
            if spec['native_box'] is None:
                pixels=bytearray(b'\xff')*(ow*oh*3);left,top=(ow-rw)//2,(oh-rh)//2
                for y in range(rh):pixels[((top+y)*ow+left)*3:((top+y)*ow+left+rw)*3]=resized[y*rw*3:(y+1)*rw*3]
                resized=bytes(pixels)
            path=directory/(spec['origin']+'.rgb');png=path.with_suffix('.png');path.write_bytes(resized)
            subprocess.run([str(a.encoder),'--from-rgb',str(ow),str(oh),str(path),str(png)],check=True,stdout=subprocess.PIPE)
            with Image.open(png) as im:
                im.load()
                if im.size!=(ow,oh) or im.convert('RGB').tobytes()!=resized:raise ValueError('independent encoded readback differs')
            candidate=dict(spec,path=str(png),sha256=sha(png),width=ow,height=oh,working_source_bounds=bounds,rgb_sha256=sha(path));candidates.append(candidate);readbacks.append(candidate)
        case=dict(case_id=caseid,id=sid,role=role,family_id=source['family_id'],cell=cell,ratio=ratio,mode=mode,width=w,height=h,image=source['image'],candidates=candidates)
        cases.append(case)
        for presentation,model in [('sol-first','gpt-6.1-sol'),('terra-first','gpt-6.1-terra'),('sol-repeat','gpt-6.1-sol')]:
            ordered=sorted(candidates,key=lambda c:hashlib.sha256((caseid+'|'+presentation+'|'+c['origin']).encode()).hexdigest());shown=[dict(c,id=letter) for letter,c in zip('ABCD',ordered)]
            task=dict(case,id=caseid+'-'+presentation,presentation_id=presentation,model=model,candidates=shown,training_eligible=False)
            path=tasks/(task['id']+'.json');path.write_text(json.dumps(task,indent=2)+'\n');requests.append(str(path))
    protocol=dict(cases=cases,task_paths=requests,readbacks=readbacks,source_roles_changed=False,test_inference=False,training_eligible=False,
        native_cache_sha256=sha(a.attention/'cache/cache.pt'),renderer=dict(resizer_sha256=sha(a.resizer),png_encoder_sha256=sha(a.encoder)),
        crop_rasterization='outward working-RGB bounds; owned resize to exact target dimensions; contain retains every working source pixel before resize')
    (a.root/'plan.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),requests=len(requests),crops=len(readbacks),fit=sum(c['role']=='fit' for c in cases),dev=sum(c['role']=='dev' for c in cases))))


if __name__=='__main__':main()
