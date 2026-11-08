#!/usr/bin/env python3
"""Paired family summaries of saved extent predictions, with no new inference."""
import argparse,json,random,statistics
from collections import defaultdict
from pathlib import Path

def paired(left,right):
 def units(rows):
  by=defaultdict(list)
  for r in rows:by[r['family_id']].append(r['iou'])
  return {f:statistics.mean(v) for f,v in by.items()}
 l,r=units(left),units(right)
 if l.keys()!=r.keys():raise ValueError('paired extent families differ')
 differences=[l[f]-r[f] for f in sorted(l)];rng=random.Random(31);samples=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(5000))
 return {'mean_iou_delta':statistics.mean(differences),'families':len(differences),'family_bootstrap_95_percentile':[samples[124],samples[4874]],'selection_data':True,'confirmatory':False}

def analyze(root):
 runs=json.loads((root/'summary.json').read_text());arms=defaultdict(list);metrics=defaultdict(list)
 for run in runs:
  path=root/(run['arm']+'-seed'+str(run['seed']))/'best-predictions.json';arms[run['arm']].extend(json.loads(path.read_text()));metrics[run['arm']].append(run['best_dev'])
 result={'directory':str(root),'arms':{},'scope':'selected epochs on eight synthetic TRAIN-development families, one localizer weak target; no human crop quality or canonical holdout claim'}
 for arm,rows in arms.items():
  values={'mean_metrics':{key:statistics.mean(m[key] for m in metrics[arm]) for key in metrics[arm][0]},'families':len({r['family_id'] for r in rows})}
  if arm=='image':
   for control in ['aspect','permuted']:
    if control in arms:values['paired_vs_'+control]=paired(rows,arms[control])
  result['arms'][arm]=values
 return result,arms

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();studies={}
for label,path in [('w24-mse',a.root/'extent-fits'),('w24-giou',a.root/'extent-giou-fits'),('w24-joint-giou',a.root/'extent-joint-fits'),('focalnet-giou',a.root/'focalnet-extent/extent-giou-fits')]:
 result,arms=analyze(path);studies[label]=result
 if label=='w24-giou':w24_aspect=arms['aspect']
 if label=='focalnet-giou':
  if arms['aspect']!=w24_aspect:raise ValueError('matched aspect-only controls unexpectedly differ')
  result['matched_aspect_control_exactly_equal']=True
with a.output.open('x') as f:json.dump(studies,f,indent=2);f.write('\n')
print(json.dumps(studies,indent=2))
