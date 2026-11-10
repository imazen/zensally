#!/usr/bin/env python3
"""Build a human-review gallery from actual pixels and retained weak labels."""
import argparse,json,shutil,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rendered',type=Path);p.add_argument('output',type=Path);p.add_argument('--admitted',type=Path);p.add_argument('--predictions',type=Path);a=p.parse_args()
    admitted=json.loads(a.admitted.read_text()) if a.admitted else {'sources':[]};labels={s['id']:s for s in admitted['sources']};predictions={};actions={}
    if a.predictions:
        data=json.loads(a.predictions.read_text());data=data.get('predictions',[]) if isinstance(data,dict) else data
        predictions={p['task_id']:p['candidate_id'] for p in data}
        actions={p['task_id']:p.get('research_action','crop') for p in data}
    records=[]
    for path in sorted(a.rendered.glob('*/packet.json')):
        s=json.loads(path.read_text());row={k:s[k] for k in ['id','use','bucket','family_id','width','height','sha256','rois','canonical_url','license','provenance']};row['image']=f'rendered/{s["id"]}/source.png';row['tasks']=[];by_candidate={(c['task_id'],c['candidate_id']):c for c in labels.get(s['id'],{}).get('candidates',[])}
        for t in s['tasks']:
            task={k:t[k] for k in ['task_id','ratio','mode']};task['candidates']=[];task['student_action']=actions.get(t['task_id'])
            for c in t['candidates']:
                label=by_candidate.get((t['task_id'],c['id']));task['candidates'].append({'id':c['id'],'image':f'rendered/{s["id"]}/{c["file"]}','xywh':c['xywh'],'teacher_acceptable':label['acceptable'] if label else None,'admitted_weight':label['weight'] if label else None,'exclusions':label['exclusions'] if label else [],'student_selected':predictions.get(t['task_id'])==c['id'],'missed_proposed_faces':c['missed_proposed_faces']})
            row['tasks'].append(task)
        records.append(row)
    for name in ['gallery-data.json','index.html']:
        p=a.output/name
        if p.exists():p.rename(p.with_name(p.name+'.bak-'+str(time.time_ns())))
    (a.output/'gallery-data.json').write_text(json.dumps({'schema':1,'human_rated':False,'sources':records},separators=(',',':'))+'\n');shutil.copyfile(Path(__file__).with_name('fit_gallery.html'),a.output/'index.html')
    print(json.dumps({'sources':len(records),'tasks':sum(len(s['tasks']) for s in records),'student_predictions':len(predictions)}))
if __name__=='__main__':main()
