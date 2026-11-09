#!/usr/bin/env python3
"""Run a fixed annotation list with bounded network concurrency and one ledger."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from api_budget import guarded
from subject_zones import build,validate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['tasks','cache','budget','authorization','output']:p.add_argument(key,type=Path)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    p.add_argument('--protocol',choices=['subject-zones','crop-corridors','label-quality'],default='subject-zones');a=p.parse_args()
    request_builder,response_validator=build,validate
    if a.protocol=='crop-corridors':
        from propose_crop_corridors import build as request_builder,validate as response_validator
    elif a.protocol=='label-quality':
        from label_quality import build as request_builder,validate as response_validator
    paths=[Path(v) for v in json.loads(a.tasks.read_text())];tasks=[json.loads(p.read_text()) for p in paths]
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('distinct source tasks required')
    for task in tasks:request_builder(task) # rejects TEST and changed image bytes before any paid call
    a.output.mkdir(parents=True,exist_ok=False);results=[]
    def call(task):
        result=guarded(task,request_builder(task),response_validator,a.budget,a.cache,a.authorization)
        path=a.output/(task['id'].replace(':','-')+'.json')
        with path.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        return dict(id=task['id'],role=task['role'],result=str(path),**result)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(call,task):task for task in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:result=future.result()
            except Exception as error:
                result=dict(id=task['id'],role=task['role'],error=type(error).__name__+': '+str(error),annotation_valid=False)
            results.append(result)
            status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),completed=len(results),required=len(tasks),
                valid=sum(r['annotation_valid'] for r in results),results=results)
            (a.output/'PROGRESS.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k in ['id','annotation_valid','new_charge_usd','error']}),flush=True)
    status.update(pid=None,stage='complete',errors=sum('error' in r for r in results))
    (a.output/'PROGRESS.json').write_text(json.dumps(status,indent=2)+'\n')
    if status['errors']:raise SystemExit(1)

if __name__=='__main__':main()

