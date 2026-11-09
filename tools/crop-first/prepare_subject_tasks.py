#!/usr/bin/env python3
"""Prepare original/owned-preview annotation contracts and independent readbacks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
from subject_zones import RATIOS


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def readback(png,rgb,reader):
    result=subprocess.run([str(reader),str(png),str(rgb)],capture_output=True,text=True,check=True)
    with Image.open(png) as image:
        if image.convert('RGB').tobytes()!=rgb.read_bytes():raise ValueError('independent source/preview RGB differs')
        return dict(size=list(image.size),rgb_sha256=digest(rgb),independent_exact=True,reader_output=result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','resizer','reader','writer','output']:p.add_argument(name,type=Path)
    p.add_argument('--policy',type=Path)
    p.add_argument('--storage-budget-mib',type=int,choices=[64,384],default=64)
    a=p.parse_args()
    if (digest(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or
            digest(a.reader)!='3346ccde58afc455194c22397ff3a5b025e4ceede6115f6795a0f8ad0c694c16' or
            digest(a.writer)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d'):
        raise ValueError('qualified owned pixel pipeline differs')
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    purposes={'two-models-tote-white':('catalogue','Feature the entire ochre tote and useful branding. Its wearer is primary; the friend is optional context.'),
        'person-speakers-white':('catalogue','Feature the complete speaker pair and product carton; the person is optional context.'),
        'friends-camera-context':('hero','Feature the foreground camera-bag wearer and product interaction; background friends may be excluded wholly.'),
        'birthday-choice-candid':('editorial','Feature the birthday guest of honor and cake/story; choose a coherent person or subgroup when necessary.')}
    policy=json.loads(a.policy.read_text()) if a.policy else None
    for job in json.loads((a.source/'plan.json').read_text()):
        identity=job['id'];source=a.source/(identity+'.png');record=json.loads((a.source/(identity+'.json')).read_text())
        if digest(source)!=record['png_sha256']:raise ValueError('native generated source differs')
        work=a.output/identity;work.mkdir();original=readback(source,work/'native.rgb',a.reader)
        if original['size']!=list(map(int,job['size'].split('x'))):raise ValueError('native dimensions differ')
        w,h=original['size'];scale=min(1,768/max(w,h));ow,oh=max(1,round(w*scale)),max(1,round(h*scale))
        with (work/'native.rgb').open('rb') as stream:
            raw=subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),','.join(map(str,[0,0,w,h,ow,oh]))],stdin=stream)
        if len(raw)!=ow*oh*3:raise ValueError('owned resized RGB length differs')
        (work/'resized.rgb').write_bytes(raw);preview=work/'preview.png'
        result=subprocess.run([str(a.writer),'--from-rgb',str(ow),str(oh),str(work/'resized.rgb'),str(preview)],capture_output=True,text=True)
        (work/'writer.log').write_text(result.stdout+result.stderr);result.check_returncode()
        reduced=readback(preview,work/'preview.rgb',a.reader)
        if (work/'preview.rgb').read_bytes()!=raw:raise ValueError('owned PNG transport changed resized samples')
        if policy is None:purpose,intent=purposes[identity];role='screening'
        else:
            item=policy[identity];purpose,intent,role=item['purpose'],item['subject_intent'],item['role']
            if role!=job['registered_role'] or role not in ['fit','dev'] or purpose not in ['catalogue','hero','detail','editorial']:raise ValueError('registered role/purpose policy differs')
        tasks=[]
        for variant,image in [('native',source),('preview',preview)]:
            task=dict(id='sunburst-v4:'+identity,role=role,width=original['size'][0],height=original['size'][1],purpose=purpose,
                subject_intent=intent,ratios=RATIOS,image=dict(path=str(image),sha256=digest(image)),
                source_png_sha256=digest(source),scenario_family=job['scenario_family'],input_variant=variant,training_eligible=False)
            path=work/('task-'+variant+'.json');path.write_text(json.dumps(task,indent=2)+'\n');tasks.append(str(path))
        rows.append(dict(id=identity,purpose=purpose,native=original,preview=reduced,tasks=tasks))
    if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>a.storage_budget_mib*1024**2:raise ValueError('explicit task storage allowance exceeded')
    (a.output/'summary.json').write_text(json.dumps(dict(sources=len(rows),readbacks=len(rows)*2,rows=rows),indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows),independent_readbacks=len(rows)*2)))


if __name__=='__main__':main()
