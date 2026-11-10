#!/usr/bin/env python3
"""Evaluate saved matched heads and bootstrap paired development families."""
import argparse,hashlib,json,random,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from fit_heads import predictions

def grades(predictions,sources):
 lookup={(s['id'],c['task_id'],c['candidate_id']):c for s in sources for c in s['candidates']};out=[]
 for p in predictions:
  c=lookup[p['source_id'],p['task_id'],p['raw_rank_candidate_id']];crop=p['research_action']=='crop';judged=c['weight']>0;positive=bool(c['acceptable'])
  out.append({'source':p['source_id'],'family':p['family_id'],'ratio':p['ratio'],'essential':1/3<=c['aspect']<=3,'requests':1,'offered':int(crop),'judged_offered':int(crop and judged),'accepted':int(crop and judged and positive),'rank_accepted':int(judged and positive),'rank_judged':int(judged)})
 return out

def summary(rows):
 counts={k:sum(r[k] for r in rows) for k in ['requests','offered','judged_offered','accepted','rank_accepted','rank_judged']}
 return {'counts':counts,'crop_coverage':counts['offered']/counts['requests'],'accepted_yield':counts['accepted']/counts['requests'],'acceptance_precision':counts['accepted']/counts['judged_offered'] if counts['judged_offered'] else None,'rank_acceptance':counts['rank_accepted']/counts['rank_judged'] if counts['rank_judged'] else None}

def paired(left,right):
 def units(rows):
  by=defaultdict(list)
  for r in rows:by[r['family']].append(r)
  return {f:summary(rs)['accepted_yield'] for f,rs in by.items()}
 l,r=units(left),units(right)
 if l.keys()!=r.keys():raise ValueError('paired families differ')
 values=[l[f]-r[f] for f in sorted(l)];rng=random.Random(31);samples=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000)];samples.sort()
 return {'mean_accepted_yield_delta':statistics.mean(values),'family_bootstrap_95_percentile':[samples[124],samples[4874]],'families':len(values),'selection_data':True,'confirmatory':False}

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();cache=torch.load(a.root/'feature-cache/features.pt',map_location='cpu',weights_only=False);indices=[i for i,u in enumerate(cache['uses']) if u=='dev'];initial=grades(predictions(cache,indices,cache['initial_logits'][indices]),cache['sources']);runs=json.loads((a.root/'head-fits/summary.json').read_text());arms=defaultdict(list);losses=defaultdict(list)
for run in runs:
 directory=a.root/'head-fits'/(run['arm']+'-seed'+str(run['seed']));arms[run['arm']].extend(grades(json.loads((directory/'best-predictions.json').read_text()),cache['sources']));losses[run['arm']].append(run['best_dev_loss'])
result={'initial':summary(initial),'initial_dev_loss':runs[0]['initial_dev']['loss'],'scope':'TRAIN development, selected epochs, noisy teacher labels; no canonical holdout or human quality claim','arms':{}}
for arm,rows in arms.items():
 result['arms'][arm]={'mean_selected_dev_loss':statistics.mean(losses[arm]),'selected_dev_losses':losses[arm],'all':summary(rows),'essential':summary([r for r in rows if r['essential']]),'ratios':{ratio:summary([r for r in rows if r['ratio']==ratio]) for ratio in ['1:3','3:1']},'paired_essential_vs_continued_baseline':paired([r for r in rows if r['essential']],[r for r in arms['baseline'] if r['essential']]),'paired_essential_vs_frozen_initializer':paired([r for r in rows if r['essential']],[r for r in initial if r['essential']])}
with (a.root/'head-analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
