#!/usr/bin/env python3
"""Register Sol-primary references, preserving roles and exact cached requests."""
import argparse
from collections import Counter
import json
from pathlib import Path
from subject_zones import build
from direct_zones import canonical,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    root=a.run;paths=[Path(v) for v in json.loads((root/'real-tasks/tasks.json').read_text())]
    paths +=[Path(v) for v in json.loads((root/'expansion-tasks/tasks.json').read_text())]
    paths +=sorted((root/'pilot-tasks-owned-rgb').glob('*/task-preview.json'))
    tasks=[dict(json.loads(path.read_text()),model='gpt-6.1-sol') for path in paths]
    if len(tasks)!=104 or len({t['id'] for t in tasks})!=104:raise ValueError('registered72real plus32synthetic sources required')
    a.output.mkdir(parents=True,exist_ok=False);out=[];requests=[]
    for task in tasks:
        request=build(task);key=sha(canonical(dict(cache_protocol=1,request=request)))
        target=a.output/(task['id'].replace(':','-')+'.json');target.write_text(json.dumps(task,indent=2)+'\n');out.append(str(target))
        requests.append(dict(id=task['id'],role=task['role'],key=key,cache_available=(root/'vision-cache'/key/'hashes.json').exists(),task_sha256=sha(target.read_bytes())))
    (a.output/'tasks.json').write_text(json.dumps(out,indent=2)+'\n')
    protocol=dict(sources=104,roles=dict(Counter(t['role'] for t in tasks)),model='gpt-6.1-sol',source_roles_changed=False,test_inputs=False,
        exact_cache_reuse=sum(r['cache_available'] for r in requests),uncached_requests=sum(not r['cache_available'] for r in requests),
        monetary_ledger=str(root/'testing/budget.jsonl'),additional_total_ceiling_usd='20',testing_subceiling_usd='8',
        owner_steering='you can do sol api calls if terra is unreliable, just track spending',
        primary_teacher_change=True,terra_caches_retained=True,annotations_still_require_geometry_and_uncertainty_audit=True,requests=requests)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:v for k,v in protocol.items() if k!='requests'}))

if __name__=='__main__':main()

