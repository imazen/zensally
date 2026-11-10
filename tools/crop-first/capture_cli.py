#!/usr/bin/env python3
"""Extract only requested weak labels/usage from an authorized CLI session.

Never copies developer instructions, auth material or full session context.
"""
import argparse,hashlib,json
from pathlib import Path
from cli_records import read_records
from teacher import parse_label,parse_selection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollout',type=Path);p.add_argument('tasks',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows=read_records(a.rollout)
    complete=[(i,r['payload']) for i,r in enumerate(rows) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='task_complete']
    if len(complete)<2:raise ValueError('expected a prior completed turn and a labeling turn')
    def usage_before(i):
        return next(r['payload']['info']['total_token_usage'] for r in reversed(rows[:i]) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='token_count')
    i,done=complete[-1];before=usage_before(complete[-2][0]);after=usage_before(i)
    labels=json.loads(done['last_agent_message']);tasks=json.loads(a.tasks.read_text())
    if [t['task_id'] for t in tasks]!=[l['task_id'] for l in labels]:raise ValueError('missing, extra or reordered task labels')
    records=[]
    for task,raw in zip(tasks,labels):
        value={k:v for k,v in raw.items() if k!='task_id'}
        pool=all('display_id' in c for c in task['candidates'])
        display_ids=[c['display_id'] for c in task['candidates']] if pool else ['A','B']
        if not pool and len(task['candidates'])!=2:raise ValueError('legacy pairs require two candidates')
        label=parse_selection(value,display_ids) if pool else parse_label({'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]})
        for image in [task['source'],*task['candidates']]:
            if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest()!=image['sha256']:raise ValueError('task pixels changed')
        records.append({'task_id':task['task_id'],'source_id':task['source_id'],'source_role':task['source_role'],
            'display_map':dict(zip(display_ids,[x['id'] for x in task['candidates']])),
            'image_sha256':[x['sha256'] for x in [task['source'],*task['candidates']]],
            'label':label,'admission':'unreviewed','training_eligible':False,'face_safety_certified':False})
    # Save only the coordinator's final explicit annotation prompt, never earlier rules.
    candidates=[]
    for row in rows[complete[-2][0]+1:i]:
        payload=row.get('payload',{})
        if row['type']=='response_item' and payload.get('role')=='user':
            candidates.extend(c['text'] for c in payload.get('content',[]) if c.get('type')=='input_text' and c['text'].startswith('New bounded vision-annotation task'))
    if len(candidates)!=1:raise ValueError('expected one bounded annotation prompt in this turn')
    texts=candidates
    context=next(r['payload'] for r in reversed(rows[:i]) if r['type']=='turn_context')
    probe=parse_label({'status':'completed','output':[{'content':[{'type':'output_text','text':complete[0][1]['last_agent_message']}]}]})
    (a.output.parent/'capability-probe.json').write_text(json.dumps({'source_role':'evaluation','purpose':'protocol-probe','training_eligible':False,'human_calibration':False,'model':context['model'],'label':probe,'usage_including_cli_context':usage_before(complete[0][0])},indent=2)+'\n')
    result={'model':context['model'],'transport':'signed-in Codex CLI, not direct Responses API',
        'task_id':done['turn_id'],'supervision':'vlm_preference','human_calibration':False,
        'usage_including_repeated_cli_context':{k:after[k]-before[k] for k in after},
        'response_text':done['last_agent_message'],'prompt':texts[0],'prompt_sha256':hashlib.sha256(texts[0].encode()).hexdigest(),'records':records}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'saved_weak_labels':len(records),'all_unreviewed':True,'model':result['model']}))


if __name__=='__main__':main()
