#!/usr/bin/env python3
"""Freeze exact old descriptors and compact teacher features on fit/dev only."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from crop_fit import CropDataset,TrainingModel,MODEL_ARGS
from composition_features import candidate_features,NAMES

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();a.output=a.root/'feature-cache';a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(31)
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inputs=json.loads((a.root/'inputs.json').read_text());manifest=Path(inputs['admitted_path'])
 if sha(manifest)!=inputs['admitted_sha256']:raise ValueError('label manifest changed')
 if sha(a.checkpoint)!='910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':raise ValueError('registered encoder/head initializer differs')
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=TrainingModel();model.load_state_dict(state['model'],strict=True);model.to(a.device);model.eval();captured=[]
 hook=model.ranker.scorer[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
 ids=[];uses=[];vectors=[];raws=[];extra=[];packed=[];sources=[];identities=[]
 for use in ['fit','dev']:
  dataset=CropDataset(manifest,use,training=use=='fit')
  for i,s in enumerate(dataset.sources):
   row=dataset[i];args={k:row[k][None].to(a.device) for k in MODEL_ARGS};raw=model(**args)
   if len(captured)!=1:raise ValueError('unexpected descriptor hook count')
   vector=captured.pop();replayed=model.ranker.scorer(vector);captured.clear()
   if not torch.equal(raw,replayed):raise ValueError('descriptor replay differs from frozen scorer')
   pose_path=a.root/'geometry'/(s['id']+'-pose.json');gaze_path=a.root/'gaze-all'/(s['id']+'.json');scan_path=a.root/'scan-all'/(s['id']+'.json')
   documents=[json.loads(path.read_text()) for path in [pose_path,gaze_path,scan_path]]
   if any(d['id']!=s['id'] or d['role']!=use or d['family_id']!=s['family_id'] for d in documents):raise ValueError('teacher sidecar identity/role differs')
   if len({d['rgb_sha256'] for d in documents})!=1:raise ValueError('teacher source RGB differs')
   features=candidate_features(row['boxes'].numpy(),*(d['result'] for d in documents));ids.append(s['id']);uses.append(use);vectors.append(vector[0].cpu());raws.append(raw[0].cpu());extra.append(torch.from_numpy(features));packed.append({k:v for k,v in row.items() if k not in ['image','index']});sources.append(s)
   identities.append({'id':s['id'],'teacher_files':{str(path):sha(path) for path in [pose_path,gaze_path,scan_path]}})
   print(json.dumps({'cached':len(ids),'id':s['id'],'role':use}),flush=True)
 hook.remove()
 cache={'ids':ids,'uses':uses,'descriptors':torch.stack(vectors),'initial_logits':torch.stack(raws),'extras':torch.stack(extra),'packed':{k:torch.stack([p[k] for p in packed]) for k in packed[0]},'sources':sources,'scorer_state':model.ranker.scorer.cpu().state_dict(),'feature_names':list(NAMES),'manifest_sha256':sha(manifest),'checkpoint_sha256':sha(a.checkpoint),'sidecar_identities':identities,'numerical_scope':'FP32 frozen feature extraction, TF32 disabled; exact descriptor->old scorer replay verified. Historical BF16 logits not substituted.'}
 torch.save(cache,a.output/'features.pt');metadata={k:v for k,v in cache.items() if k not in ['descriptors','initial_logits','extras','packed','sources','scorer_state']};metadata.update({'sources':len(ids),'descriptor_shape':list(cache['descriptors'].shape),'extra_shape':list(cache['extras'].shape),'payload_bytes':sum(t.numel()*t.element_size() for t in [cache['descriptors'],cache['initial_logits'],cache['extras'],*cache['packed'].values()]),'cache_sha256':sha(a.output/'features.pt')})
 (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k!='sidecar_identities'}))
if __name__=='__main__':main()
