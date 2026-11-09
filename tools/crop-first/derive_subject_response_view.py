#!/usr/bin/env python3
"""Derive a complete source-bound response view without making API requests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from direct_zones import sha
from subject_zones import validate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['tasks','responses','output']:p.add_argument(key,type=Path)
    p.add_argument('--source-prefix',required=True);p.add_argument('--expected',type=int,required=True);a=p.parse_args()
    parent=json.loads((a.responses/'PROGRESS.json').read_text())
    if parent.get('stage')!='complete' or parent.get('errors') or parent['completed']!=parent['required']:raise ValueError('complete parent response pass required')
    paths=[Path(v) for v in json.loads(a.tasks.read_text())];selected=[(path,json.loads(path.read_text())) for path in paths if json.loads(path.read_text())['id'].startswith(a.source_prefix)]
    if len(selected)!=a.expected:raise ValueError('registered response-view source count differs')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'tasks').mkdir();tasks=[];results=[]
    for path,task in selected:
        name=task['id'].replace(':','-')+'.json';result_path=a.responses/name
        result=json.loads(result_path.read_text());entry=Path(result['entry']);metadata=json.loads((entry/'metadata.json').read_text())
        if metadata['source_id']!=task['id'] or metadata['source_role']!=task['role']:raise ValueError('source role/identity differs')
        for name_,digest in json.loads((entry/'hashes.json').read_text()).items():
            if sha((entry/name_).read_bytes())!=digest:raise ValueError('immutable parent cache differs')
        try:validate(json.loads((entry/'response.json').read_text()),task);valid=True
        except (ValueError,KeyError,TypeError):valid=False
        if valid!=result['annotation_valid']:raise ValueError('structural validation differs')
        target=a.output/'tasks'/path.name;shutil.copy2(path,target);tasks.append(str(target))
        shutil.copy2(result_path,a.output/name);results.append(dict(id=task['id'],role=task['role'],**result))
    (a.output/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (a.output/'PROGRESS.json').write_text(json.dumps(dict(schema='completed-response-view-v1',pid=None,stage='complete',completed=len(results),required=a.expected,
        valid=sum(r['annotation_valid'] for r in results),errors=0,results=results,new_model_api_calls=0,parent_progress_sha256=sha((a.responses/'PROGRESS.json').read_bytes())),indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps(dict(source_prefix=a.source_prefix,sources=len(results),roles=dict(Counter(t['role'] for _,t in selected)),
        source_roles_changed=False,new_model_api_calls=0,parent_tasks_sha256=sha(a.tasks),parent_responses=str(a.responses)),indent=2)+'\n')
    print(json.dumps(dict(sources=len(results),valid=sum(r['annotation_valid'] for r in results),new_model_api_calls=0)))

if __name__=='__main__':main()

