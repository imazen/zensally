#!/usr/bin/env python3
"""Extract only requested weak labels/usage from an authorized CLI session.

Never copies developer instructions, auth material or full session context.
"""
import argparse,hashlib,json
from pathlib import Path
from teacher import parse_label


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollout',type=Path);p.add_argument('tasks',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows=[json.loads(s) for s in a.rollout.read_text().splitlines()]
    complete=[(i,r['payload']) for i,r in enumerate(rows) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='task_complete']
    if len(complete)!=2:raise ValueError('expected exactly the capability probe and one bounded four-task pilot')
    def usage_before(i):
        return next(r['payload']['info']['total_token_usage'] for r in reversed(rows[:i]) if r['type']=='event_msg' and r.get('payload',{}).get('type')=='token_count')
    i,done=complete[-1];before=usage_before(complete[0][0]);after=usage_before(i)
    labels=json.loads(done['last_agent_message']);tasks=json.loads(a.tasks.read_text())
    if [t['task_id'] for t in tasks]!=[l['task_id'] for l in labels]:raise ValueError('missing, extra or reordered task labels')
    records=[]
    for task,raw in zip(tasks,labels):
        label=parse_label({'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps({k:v for k,v in raw.items() if k!='task_id'})}]}]})
        for image in [task['source'],*task['candidates']]:
            if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest()!=image['sha256']:raise ValueError('task pixels changed')
        records.append({'task_id':task['task_id'],'source_id':task['source_id'],'source_role':task['source_role'],
            'display_map':dict(zip(['A','B'],[x['id'] for x in task['candidates']])),
            'image_sha256':[x['sha256'] for x in [task['source'],*task['candidates']]],
            'label':label,'admission':'unreviewed','training_eligible':False,'face_safety_certified':False})
    # Save only the coordinator's final explicit annotation prompt, never earlier rules.
    prompt=next(r['payload'] for r in reversed(rows[:i]) if r['type']=='response_item' and r.get('payload',{}).get('role')=='user')
    texts=[c['text'] for c in prompt.get('content',[]) if c.get('type')=='input_text']
    if not texts or not texts[0].startswith('New bounded vision-annotation task'):raise ValueError('unexpected prompt')
    context=next(r['payload'] for r in reversed(rows[:i]) if r['type']=='turn_context')
    probe=parse_label({'status':'completed','output':[{'content':[{'type':'output_text','text':complete[0][1]['last_agent_message']}]}]})
    (a.output.parent/'capability-probe.json').write_text(json.dumps({'source_role':'evaluation','purpose':'protocol-probe','training_eligible':False,'human_calibration':False,'model':context['model'],'label':probe,'usage_including_cli_context':before},indent=2)+'\n')
    result={'model':context['model'],'transport':'signed-in Codex CLI, not direct Responses API',
        'task_id':done['turn_id'],'supervision':'vlm_preference','human_calibration':False,
        'usage_including_repeated_cli_context':{k:after[k]-before[k] for k in after},
        'prompt':texts[0],'prompt_sha256':hashlib.sha256(texts[0].encode()).hexdigest(),'records':records}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'saved_weak_labels':len(records),'all_unreviewed':True,'model':result['model']}))


if __name__=='__main__':main()
