#!/usr/bin/env python3
"""Recover immutable epoch predictions from retained weights; never refit or replace evidence."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from fit_heads import ARMS,evaluate,predictions,sha
from composition_head import CompositionHead

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.05);torch.backends.cuda.matmul.allow_tf32=False
 path=a.root/'feature-cache/features.pt';cache=torch.load(path,map_location='cpu',weights_only=False);identity=sha(path);dev=[i for i,u in enumerate(cache['uses']) if u=='dev'];data={'descriptors':cache['descriptors'].cuda(),'packed':{k:v.cuda() for k,v in cache['packed'].items()}};extra=cache['extras'].cuda();scorer=nn.Sequential(nn.Linear(762,96),nn.ReLU(),nn.Linear(96,2));scorer.load_state_dict(cache['scorer_state']);count=0;best_checked=0
 for directory in sorted((a.root/'head-fits').glob('*-seed*')):
  summary=json.loads((directory/'summary.json').read_text());best=json.loads((directory/'best-predictions.json').read_text())
  for checkpoint in sorted(directory.glob('epoch-*.pt')):
   state=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if state['cache_sha256']!=identity:raise ValueError('checkpoint/cache mismatch')
   model=CompositionHead(scorer,ARMS[state['arm']]).cuda().eval();model.load_state_dict(state['model']);features=extra[state['protocol']['source_permutation']] if state['arm']=='permuted' else extra
   _,raw=evaluate(model,cache,dev,data,features);result=predictions(cache,dev,raw);output=directory/f"predictions-epoch-{state['epoch']:03}.json"
   if output.exists():
    if json.loads(output.read_text())!=result:raise ValueError('existing epoch predictions disagree')
   else:
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
   if state['epoch']==summary['best_epoch']:
    if result!=best:raise ValueError('recovered selected predictions differ from retained result')
    best_checked+=1
   count+=1
 report={'recovered_checkpoint_predictions':count,'selected_best_exactly_reproduced':best_checked,'fitting_repeated':False,'selection_changed':False,'original_best_json_preserved':True,'cache_sha256':identity}
 with (a.root/'head-fits/prediction-recovery.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
