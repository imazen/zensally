#!/usr/bin/env python3
"""Fixed research abstention policy from saved logits; no threshold fitting."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=a.run
    manifest=json.loads((r/'admitted.json').read_text());lookup={};tasks={};faces={}
    for s in manifest['sources']:
        for c in s['candidates']:lookup[(c['task_id'],c['candidate_id'])]=c;tasks.setdefault(c['task_id'],[]).append(c)
    for p in (r/'rendered').glob('*/packet.json'):
        for t in json.loads(p.read_text())['tasks']:
            for c in t['candidates']:faces[(t['task_id'],c['id'])]=c['missed_proposed_faces']
    result={}
    for use,path in [('dev',r/'pretrained-seed31/best-predictions.json'),('eval',r/'canonical-eval/evaluation.json')]:
        data=json.loads(path.read_text());predictions=data.get('predictions',[]) if isinstance(data,dict) else data;stats=defaultdict(Counter);ratios=defaultdict(Counter);suggestions=[]
        for pred in predictions:
            c=lookup[(pred['task_id'],pred['candidate_id'])];tier='essential' if 1/3<=c['aspect']<=3 else 'secondary';crop=pred['acceptance_logit']>=0 and not faces[(pred['task_id'],pred['candidate_id'])]
            for out in [stats[tier],ratios[c['ratio']]]:
                out['tasks']+=1;out['suggest_crop']+=int(crop)
                if crop and pred['judged']:out['crop_judged']+=1;out['crop_acceptable']+=int(pred['teacher_acceptable'])
                if crop and not pred['judged']:out['crop_unjudged']+=1
            suggestions.append({**pred,'raw_rank_candidate_id':pred['candidate_id'],'research_action':'crop' if crop else 'contain','candidate_id':pred['candidate_id'] if crop else None,'reason':'fixed 0.5 acceptance threshold and proposed-face geometry veto; trusted face coverage remains unverified'})
        result[use]={'tiers':{k:dict(v) for k,v in stats.items()},'ratios':{k:dict(v) for k,v in ratios.items()}}
        with (a.output/(use+'-suggestions.json')).open('x') as f:json.dump(suggestions,f,indent=2)
    with (a.output/'metrics.json').open('x') as f:json.dump({'policy':'fixed top-rank acceptance threshold 0.5 and proposed-face veto; research only','canonical_eval_used_to_choose_threshold':False,'human_rated':False,'metrics':result},f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
