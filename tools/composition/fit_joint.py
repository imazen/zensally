#!/usr/bin/env python3
"""Registered resumable end-to-end geometry distillation, with matched controls."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch import nn
from crop_fit import CropDataset,ResidentBatches,TrainingModel,MODEL_ARGS,loss
from composition_joint import AuxiliaryScorer,auxiliary_loss
from fit_heads import predictions,sha

def retain(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def alias(path,target):
 stage=path.with_name(path.name+'.new');stage.symlink_to(target.name);stage.replace(path)

@torch.no_grad()
def evaluate(model,dev,cache,aux,aux_indices):
 model.eval();batch=next(iter(dev));raw=model(**{k:batch[k] for k in MODEL_ARGS});value,accept,rank=loss(raw,batch);indices=[i for i,u in enumerate(cache['uses']) if u=='dev'];result=predictions(cache,indices,raw[...,:2].cpu());ai=aux_indices
 aux_raw=model(image=aux['image'][ai],**{k:aux['packed'][k][ai] for k in MODEL_ARGS if k!='image'});mask=aux['packed']['mask'][ai];target=aux['packed']['targets'][ai];error=(aux_raw[...,2:]-target).abs();groups={}
 for name,start,end in [('pose',0,16),('gaze',16,24),('scan',24,32)]:
  m=mask[...,start:end];groups[name]={'mae':float((error[...,start:end]*m).sum()/m.sum().clamp_min(1)),'observations':int(m.sum())}
 return {'loss':float(value),'accept_loss':float(accept),'rank_loss':float(rank),'geometry':groups},result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('first_gate',type=Path);p.add_argument('checkpoint',type=Path);a=p.parse_args();out=a.root/'joint-fits';out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 aux_path=a.root/'auxiliary.pt';aux=torch.load(aux_path,map_location='cpu',weights_only=False);aux_identity=sha(aux_path);cache=torch.load(a.first_gate/'feature-cache/features.pt',map_location='cpu',weights_only=False);manifest=Path(json.loads((a.first_gate/'inputs.json').read_text())['admitted_path'])
 if sha(a.checkpoint)!='910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':raise ValueError('registered initializer changed')
 fit=ResidentBatches(CropDataset(manifest,'fit',training=True),16,'cuda',128*1024**2,shuffle=True);dev=ResidentBatches(CropDataset(manifest,'dev'),16,'cuda',32*1024**2)
 if sha(manifest)!=cache['manifest_sha256']:raise ValueError('crop manifest changed')
 ai_fit=[i for i,s in enumerate(aux['sources']) if s['use']=='fit'];ai_dev=[i for i,s in enumerate(aux['sources']) if s['use']=='dev'];permutation=list(range(56));rng=torch.Generator().manual_seed(31)
 for indices in [ai_fit,ai_dev]:
  order=torch.randperm(len(indices),generator=rng).tolist()
  for i,j in enumerate(order):permutation[indices[i]]=indices[j]
 aux['image']=aux['image'].cuda().contiguous(memory_format=torch.channels_last);aux['packed']={k:v.cuda() for k,v in aux['packed'].items()};protocol={'arms':['baseline','geometry','permuted'],'seeds':[31,137,911],'epochs':40,'batch':16,'lr':.00003,'weight_decay':.01,'aux_weight':.1,'clip':5.,'precision':'FP32 TF32 disabled','batchnorm':'frozen running statistics, trainable affine','selection':'minimum original TRAIN-dev loss including epoch zero','auxiliary_sha256':aux_identity,'manifest_sha256':sha(manifest),'checkpoint_sha256':sha(a.checkpoint),'source_permutation':permutation,'canonical_holdouts_opened':False,'composition_labels_from_new_AI':False}
 if (out/'protocol.json').exists():
  if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('resume protocol differs')
 else:retain(out/'protocol.json',protocol)
 reports=[]
 for arm in protocol['arms']:
  for seed in protocol['seeds']:
   directory=out/(arm+'-seed'+str(seed));directory.mkdir(exist_ok=True)
   if (directory/'summary.json').exists():reports.append(json.loads((directory/'summary.json').read_text()));continue
   torch.manual_seed(seed);model=TrainingModel();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);model.ranker.scorer=AuxiliaryScorer(model.ranker.scorer);model.cuda().to(memory_format=torch.channels_last).eval();optimizer=torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=.01,fused=True);history=[];best=float('inf');best_epoch=None;start=0
   if (directory/'latest.pt').exists():
    state=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    if state['protocol']!=protocol:raise ValueError('checkpoint protocol differs')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start=state['epoch']+1;torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
   for epoch in range(start,41):
    if epoch:
     # Eval keeps BatchNorm statistics fixed, while gradients update all parameters.
     model.eval();fit.seed=seed;fit.epoch=epoch-1
     for batch in fit:
      optimizer.zero_grad(set_to_none=True);raw=model(**{k:batch[k] for k in MODEL_ARGS});value,_,_=loss(raw,batch);value.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
     order=torch.randperm(48,generator=torch.Generator().manual_seed(seed+epoch-1)).tolist()
     for offset in range(0,48,16):
      indices=[ai_fit[j] for j in order[offset:offset+16]];label_indices=[permutation[i] for i in indices] if arm=='permuted' else indices;optimizer.zero_grad(set_to_none=True)
      with torch.set_grad_enabled(arm!='baseline'):
       raw=model(image=aux['image'][indices],**{k:aux['packed'][k][indices] for k in MODEL_ARGS if k!='image'});value=.1*auxiliary_loss(raw,aux['packed']['targets'][label_indices],aux['packed']['mask'][label_indices])
      if arm!='baseline':value.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    metrics,preds=evaluate(model,dev,cache,aux,ai_dev);history.append({'epoch':epoch,'dev':metrics});improved=metrics['loss']<best
    if improved:best=metrics['loss'];best_epoch=epoch
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()};checkpoint=directory/f'epoch-{epoch:03}.pt';torch.save(state,checkpoint);alias(directory/'latest.pt',checkpoint);prediction_path=directory/f'predictions-epoch-{epoch:03}.json';retain(prediction_path,preds)
    if improved:alias(directory/'best.pt',checkpoint);alias(directory/'best-predictions.json',prediction_path)
    (out/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':'joint geometry fitting','arm':arm,'seed':seed,'epoch':epoch,'required_epochs':40,'completed_runs':len(reports),'required_runs':9,'checked_unix':time.time()}))
    if epoch%10==0:print(json.dumps({'arm':arm,'seed':seed,'epoch':epoch,'metrics':metrics,'best_epoch':best_epoch}),flush=True)
   report={'arm':arm,'seed':seed,'best_epoch':best_epoch,'best_dev_loss':best,'parameters':sum(p.numel() for p in model.parameters()),'final_geometry':history[-1]['dev']['geometry']};retain(directory/'history.json',history);retain(directory/'summary.json',report);reports.append(report)
 retain(out/'summary.json',reports);(out/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed_runs':len(reports),'required_runs':9,'checked_unix':time.time()}));print(json.dumps({'completed_runs':len(reports),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}))
if __name__=='__main__':main()
