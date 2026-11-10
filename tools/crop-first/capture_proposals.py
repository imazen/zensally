#!/usr/bin/env python3
"""Capture one bounded source-conditioned proposal response, never full CLI context."""
import argparse,hashlib,json
from pathlib import Path
from cli_records import read_records
from proposals import candidate_pool


def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollout',type=Path);p.add_argument('batch',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rs=read_records(a.rollout);completed=[(i,r['payload']) for i,r in enumerate(rs) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='task_complete'];i,done=completed[-1]
    prompts=[]
    for row in rs[completed[-2][0]+1:i]:
        v=row.get('payload',{})
        if row['type']=='response_item' and v.get('role')=='user':
            prompts.extend(c['text'] for c in v.get('content',[]) if c.get('type')=='input_text' and c['text'].startswith('New bounded vision-proposal task.'))
    if len(prompts)!=1 or str(a.batch) not in prompts[0]:raise ValueError('response is not the requested source-proposal batch')
    text=prompts[0]
    batch=json.loads(a.batch.read_text());raw=done['last_agent_message'];labels=json.loads(raw)
    expected=[t['task_id'] for s in batch for t in s['tasks']]
    if [r['task_id'] for r in labels]!=expected:raise ValueError('missing/extra/reordered proposal tasks')
    index={t['task_id']:(s,t) for s in batch for t in s['tasks']}
    for r in labels:
        if set(r)!={'task_id','action','proposals','reason'} or r['action'] not in ['crop','contain','unchanged'] or not isinstance(r['reason'],str):raise ValueError('invalid direct proposal record')
        if not isinstance(r['proposals'],list) or len(r['proposals'])>2 or (r['action']=='contain')!=(len(r['proposals'])==0):raise ValueError('invalid proposal count/action')
        s,t=index[r['task_id']];candidate_pool(s['width'],s['height'],t['mode'],r['proposals'],target_ratio=t['target_ratio'])
    for s in batch:
        if sha(Path(s['source']['path']).read_bytes())!=s['source']['sha256']:raise ValueError('source preview changed')
    def usage_before(i):return next(r['payload']['info']['total_token_usage'] for r in reversed(rs[:i]) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='token_count')
    before=usage_before(completed[-2][0]);after=usage_before(i)
    context=next(r['payload'] for r in reversed(rs[:i]) if r['type']=='turn_context')
    evidence={'model':context['model'],'source_roles':{s['id']:s['role'] for s in batch},'source_image_sha256':{s['id']:s['source']['sha256'] for s in batch},'batch_sha256':sha(a.batch.read_bytes()),'prompt':text,'prompt_sha256':sha(text.encode()),'response_text':raw,'usage_including_cli_context':{k:after[k]-before[k] for k in after},'admission':'unreviewed','training_eligible':False,'face_safety_certified':False}
    with a.output.open('x') as f:f.write(json.dumps(labels,indent=2)+'\n')
    with a.output.with_suffix('.evidence.json').open('x') as f:f.write(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'captured_tasks':len(labels),'crop':sum(r['action']=='crop' for r in labels),'contain':sum(r['action']=='contain' for r in labels)}))


if __name__=='__main__':main()
