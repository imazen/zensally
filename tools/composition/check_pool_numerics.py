#!/usr/bin/env python3
"""Compare training-pool/precision paths on fixed fit/dev crop choices, never holdouts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from crop_fit import CropDataset,ResidentBatches,TrainingModel,MODEL_ARGS,loss
from fit_heads import predictions,sha

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first_gate',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cache=torch.load(a.first_gate/'feature-cache/features.pt',map_location='cpu',weights_only=False);manifest=Path(json.loads((a.first_gate/'inputs.json').read_text())['admitted_path'])
 if sha(manifest)!=cache['manifest_sha256'] or sha(a.checkpoint)!=cache['checkpoint_sha256']:raise ValueError('frozen identities differ')
 model=TrainingModel();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);model.cuda().to(memory_format=torch.channels_last).eval();loaders=[ResidentBatches(CropDataset(manifest,use,training=use=='fit'),16,'cuda',128*1024**2) for use in ['fit','dev']];reference=None;summary={}
 for name,pool,precision in [('fp32-integral','integral','fp32'),('fp32-matmul','matmul','fp32'),('bf16-integral','integral','bf16'),('bf16-matmul','matmul','bf16')]:
  model.ranker.pool_backend=pool;raws=[]
  for loader in loaders:
   for batch in loader:
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):raw=model(**{k:batch[k] for k in MODEL_ARGS})
    if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite numerical path')
    raws.append(raw.float().cpu())
  raw=torch.cat(raws);pred=predictions(cache,list(range(80)),raw);report={'max_abs_logit_error':0.,'roles':{}}
  if reference is None:reference=(raw,pred)
  else:report['max_abs_logit_error']=float((raw-reference[0]).abs().max())
  by_key={(r['source_id'],r['task_id']):r for r in reference[1]}
  for use in ['fit','dev']:
   ids={s['id'] for i,s in enumerate(cache['sources']) if cache['uses'][i]==use};counts={'tasks':0,'rank_choice_changes':0,'research_action_changes':0,'raw_edge_moves_over_3_map_pixels':0,'offered_edge_moves_over_3_map_pixels':0,'max_raw_edge_move_map_pixels':0.,'max_offered_edge_move_map_pixels':0.}
   for r in pred:
    if r['source_id'] not in ids:continue
    b=by_key[r['source_id'],r['task_id']];s=next(s for s in cache['sources'] if s['id']==r['source_id']);boxes={c['candidate_id']:c['box'] for c in s['candidates']};movement=max(abs(x-y)*128 for x,y in zip(boxes[r['raw_rank_candidate_id']],boxes[b['raw_rank_candidate_id']]));offered=r['research_action']==b['research_action']=='crop';counts['tasks']+=1;counts['rank_choice_changes']+=int(r['raw_rank_candidate_id']!=b['raw_rank_candidate_id']);counts['research_action_changes']+=int(r['research_action']!=b['research_action']);counts['raw_edge_moves_over_3_map_pixels']+=int(movement>3);counts['max_raw_edge_move_map_pixels']=max(counts['max_raw_edge_move_map_pixels'],movement)
    if offered:counts['offered_edge_moves_over_3_map_pixels']+=int(movement>3);counts['max_offered_edge_move_map_pixels']=max(counts['max_offered_edge_move_map_pixels'],movement)
   report['roles'][use]=counts
  with (a.output/(name+'-predictions.json')).open('x') as f:json.dump(pred,f,indent=2);f.write('\n')
  summary[name]=report;print(json.dumps({name:report}),flush=True)
 summary['scope']='same frozen weights, batch16 channels-last FP32/TF32-off reference; raw rankings and research acceptance only, not shipping coverage certification. Three map pixels = normalized edge displacement *128. No diagnostic maps or canonical holdouts.';(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
