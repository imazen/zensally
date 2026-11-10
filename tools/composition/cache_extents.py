#!/usr/bin/env python3
"""Admit weak primary extents and cache frozen visual descriptors by fixed roles."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch.nn import functional as F
from crop_fit import TrainingModel
from fit_heads import sha

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('checkpoint',type=Path);a=p.parse_args();out=a.root/'extent-cache';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if sha(a.checkpoint)!='910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':raise ValueError('registered encoder differs')
 aux=torch.load(a.auxiliary/'auxiliary.pt',map_location='cpu',weights_only=False);model=TrainingModel();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);model.cuda().eval();visual=[];targets=[];present=[];sources=[];identity=[];geometry=[];protocol=json.loads((a.root/'primary56/protocol.json').read_text());plan=json.loads((a.root/'primary-query-plan.json').read_text())
 if protocol['queries']!=plan['queries'] or protocol['threshold']!=.3 or protocol['text_threshold']!=.25:raise ValueError('frozen teacher query/threshold protocol differs')
 for start in range(0,56,16):
  maps=model.ranker.encode(aux['image'][start:start+16].cuda().contiguous(memory_format=torch.channels_last));visual.append(torch.cat([F.adaptive_avg_pool2d(m,(4,4)).flatten(1) for m in maps],-1).cpu())
 for i,s in enumerate(aux['sources']):
  path=a.root/'primary56'/(s['id']+'.json');row=json.loads(path.read_text())
  if row['id']!=s['id'] or row['source_rgb_sha256']!=s['rgb_sha256'] or row['query']!=plan['queries'][s['id']]:raise ValueError('localizer source identity differs')
  boxes=[]
  if len(row['detections'])>900:raise ValueError('teacher exceeds query count')
  for d in row['detections']:
   b=d['visible_box_normalized']
   if len(b)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in b) or not math.isfinite(d['score']) or d['score']<.3:raise ValueError('invalid frozen teacher box/score')
   if b[2]>b[0] and b[3]>b[1]:boxes.append(b)
  target=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)] if boxes else [0,0,1,1];targets.append(target);present.append(bool(boxes));w,h=row['source_size'];valid=aux['packed']['valid_rect'][i];aspect=math.log(w/h);geometry.append([aspect,math.tanh(aspect),float(valid[2]-valid[0]),float(valid[3]-valid[1])]);sources.append({**s,'primary_extent_training_eligible':s['use']=='fit' and bool(boxes),'primary_extent_present':bool(boxes),'target':target if boxes else None,'target_origin':'union of visible text-conditioned localizer detections; weak, possibly incomplete/false-positive','human_rated':False,'face_coverage_certified':False});identity.append({'id':s['id'],'teacher_sha256':sha(path)})
 cache={'visual':torch.cat(visual),'geometry':torch.tensor(geometry,dtype=torch.float32),'targets':torch.tensor(targets,dtype=torch.float32),'present':torch.tensor(present),'sources':sources,'teacher_identities':identity,'teacher_protocol_sha256':sha(a.root/'primary56/protocol.json'),'query_plan_sha256':sha(a.root/'primary-query-plan.json'),'encoder_sha256':sha(a.checkpoint),'auxiliary_source_cache_sha256':sha(a.auxiliary/'auxiliary.pt'),'schema':1,'supervision':'noisy_localizer_primary_extents','canonical_holdouts_opened':False};torch.save(cache,out/'features.pt');metadata={k:v for k,v in cache.items() if k not in ['visual','geometry','targets','present']};metadata.update(cache_sha256=sha(out/'features.pt'),fit_eligible=sum(s['primary_extent_training_eligible'] for s in sources),dev_eligible=sum(s['use']=='dev' and s['primary_extent_present'] for s in sources));(out/'admission.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'fit':metadata['fit_eligible'],'dev':metadata['dev_eligible'],'missing':present.count(False),'cache_sha256':metadata['cache_sha256']}))
if __name__=='__main__':main()
