#!/usr/bin/env python3
"""Prepare only eligible v2 intervals and reuse the already-paid first reviews."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','native','resizer','encoder','output']:p.add_argument(key,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False);tasks=[];records=[]
    for case in json.loads((a.root/'plan.json').read_text())['cases']:
        result=a.root/'corridor-v2-results'/(case['id'].replace(':','-')+'.json');reply=json.loads(result.read_text())
        if not reply['annotation_valid']:
            records.append(dict(case_id=case['case_id'],stage='invalid proposal retained',tasks=0));continue
        zones=json.loads((Path(reply['entry'])/'zones.json').read_text());eligible=[r for r in zones['projected_ranges'] if r['ready_for_rendered_review']]
        if not eligible:
            records.append(dict(case_id=case['case_id'],stage='no admitted-for-review intervals',tasks=0));continue
        if case['case_id']=='case01':
            directory=a.root/'case01-v2-review'
        else:
            directory=a.output/case['case_id']
            subprocess.run([sys.executable,str(Path(__file__).with_name('render_crop_corridors.py')),str(a.root),case['case_id'],str(result),str(a.native),str(a.resizer),str(a.encoder),str(directory)],check=True)
        plan=json.loads((directory/'plan.json').read_text())
        if len(plan['groups'])!=len(eligible):raise ValueError('prepared interval count differs')
        tasks.extend(plan['task_paths']);records.append(dict(case_id=case['case_id'],stage='prepared',tasks=len(plan['task_paths']),plan=str(directory/'plan.json')))
    if len(tasks)!=len(set(tasks)):raise ValueError('duplicate paid task path')
    (a.output/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n');(a.output/'index.json').write_text(json.dumps(dict(sources=records,requests=len(tasks),roles_changed=False,test_inference=False,training_started=False),indent=2)+'\n')
    print(json.dumps(dict(requests=len(tasks),intervals=len(tasks)//3,source_groups=sum(r['tasks']>0 for r in records))))


if __name__=='__main__':main()
