#!/usr/bin/env python3
"""Matched frozen-encoder primary-extent heads and source-permuted controls."""
import argparse,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from composition_extent import ExtentHead,box_metrics,extent_objective
from fit_heads import sha

def retain(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def alias(path,target):
 stage=path.with_name(path.name+'.new');stage.symlink_to(target.name);stage.replace(path)

@torch.no_grad()
def evaluate(model,features,target,indices,sources,objective="mse"):
 box=model(features[indices]);metrics,iou=box_metrics(box,target[indices]);rows=[]
 if objective=='l1giou':metrics['objective']=float(extent_objective(box,target[indices]))
 if not bool(torch.isfinite(box).all()) or not bool((box[:,:2]>=0).all()&(box[:,2:]<=1).all()&(box[:,2:]>box[:,:2]).all()):raise ValueError('invalid extent output')
 for j,i in enumerate(indices):rows.append({'id':sources[i]['id'],'family_id':sources[i]['family_id'],'use':sources[i]['use'],'box':box[j].cpu().tolist(),'weak_target':target[i].cpu().tolist(),'iou':float(iou[j]),'edge_mae':float((box[j]-target[i]).abs().mean()),'face_coverage_certified':False})
 return metrics,rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--loss',choices=['mse','l1giou'],default='mse');a=p.parse_args();out=a.root/('extent-giou-fits' if a.loss=='l1giou' else 'extent-fits');out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;path=a.root/'extent-cache/features.pt';cache=torch.load(path,map_location='cpu',weights_only=False);sources=cache['sources'];fit=[i for i,s in enumerate(sources) if s['primary_extent_training_eligible']];dev=[i for i,s in enumerate(sources) if s['use']=='dev' and s['primary_extent_present']]
 if {sources[i]['family_id'] for i in fit}&{sources[i]['family_id'] for i in dev}:raise ValueError('extent source families cross roles')
 if not fit or not dev:raise ValueError('nonempty explicit weak-label roles required')
 visual=cache['visual'];mean=visual[fit].mean(0);std=visual[fit].std(0,unbiased=False).clamp_min(.001);visual=((visual-mean)/std).clamp(-10,10);geometry=cache['geometry'];target=cache['targets'].cuda();permutation=list(range(56));rng=torch.Generator().manual_seed(31);groups=[]
 for use in ['fit','dev']:
  for lower,upper in [(0,.5),(.5,.9),(.9,1.1),(1.1,2.),(2.,float('inf'))]:
   indices=[i for i,s in enumerate(sources) if s['use']==use and lower<=math.exp(float(geometry[i,0]))<upper];order=torch.randperm(len(indices),generator=rng).tolist();groups.append({'role':use,'lower':lower,'upper':upper if math.isfinite(upper) else None,'sources':indices})
   for i,j in enumerate(order):permutation[indices[i]]=indices[j]
 inputs={'aspect':torch.cat((torch.zeros_like(visual),geometry),-1).cuda(),'image':torch.cat((visual,geometry),-1).cuda(),'permuted':torch.cat((visual[permutation],geometry),-1).cuda()};protocol={'cache_sha256':sha(path),'arms':list(inputs),'seeds':[31,137,911],'epochs':200,'batch':16,'lr':.001,'weight_decay':.01,'clip':5.,'precision':'FP32 TF32 disabled','loss':'5*mean edge L1 + 2*generalized IoU' if a.loss=='l1giou' else 'source-balanced mean squared error on four normalized edges','selection':'minimum weak TRAIN-development L1/GIoU objective, including epoch zero' if a.loss=='l1giou' else 'minimum weak TRAIN-development MSE, including epoch zero','source_permutation':permutation,'permutation_groups':groups,'actually_moved_fit_sources':sum(permutation[i]!=i for i in fit),'actually_moved_dev_sources':sum(permutation[i]!=i for i in dev),'canonical_holdouts_opened':False,'trained_visual_encoder':False,'minimum_partition_width':.0001,'fit_sources':len(fit),'dev_sources':len(dev)}
 if (out/'protocol.json').exists():
  if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('resume changes registered extent protocol')
 else:retain(out/'protocol.json',protocol)
 control=target[fit].mean(0).expand(len(dev),4);metrics,iou=box_metrics(control,target[dev]);
 if a.loss=='l1giou':metrics['objective']=float(extent_objective(control,target[dev]))
 control_rows=[{'id':sources[i]['id'],'family_id':sources[i]['family_id'],'iou':float(iou[j]),'edge_mae':float((control[j]-target[i]).abs().mean()),'box':control[j].cpu().tolist(),'weak_target':target[i].cpu().tolist()} for j,i in enumerate(dev)]
 if not (out/'fit-mean-control.json').exists():retain(out/'fit-mean-control.json',{'metrics':metrics,'predictions':control_rows})
 reports=[]
 for arm,features in inputs.items():
  for seed in protocol['seeds']:
   directory=out/(arm+'-seed'+str(seed));directory.mkdir(exist_ok=True)
   if (directory/'summary.json').exists():reports.append(json.loads((directory/'summary.json').read_text()));continue
   torch.manual_seed(seed);model=ExtentHead().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,fused=True);history=[];best=float('inf');best_epoch=None;start=0
   if (directory/'latest.pt').exists():
    state=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    if state['protocol']!=protocol:raise ValueError('checkpoint protocol differs')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start=state['epoch']+1
   for epoch in range(start,201):
    if epoch:
     model.train();order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch-1)).tolist()
     for offset in range(0,len(fit),16):
      indices=[fit[j] for j in order[offset:offset+16]];optimizer.zero_grad(set_to_none=True);prediction=model(features[indices]);value=extent_objective(prediction,target[indices]) if a.loss=='l1giou' else (prediction-target[indices]).square().mean()
      if not bool(torch.isfinite(value)):raise ValueError('nonfinite extent fitting loss')
      value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    model.eval();metrics,rows=evaluate(model,features,target,dev,sources,a.loss);history.append({'epoch':epoch,'dev':metrics});improved=metrics.get('objective',metrics['edge_mse'])<best
    if improved:best=metrics.get('objective',metrics['edge_mse']);best_epoch=epoch
    if improved or epoch%25==0:
     state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best':best,'best_epoch':best_epoch,'normalization_mean':mean,'normalization_std':std};checkpoint=directory/f'epoch-{epoch:03}.pt';checkpoint_stream=checkpoint.open('xb');torch.save(state,checkpoint_stream);checkpoint_stream.close();alias(directory/'latest.pt',checkpoint);prediction_path=directory/f'predictions-epoch-{epoch:03}.json';retain(prediction_path,rows)
     if improved:alias(directory/'best.pt',checkpoint);alias(directory/'best-predictions.json',prediction_path)
    (out/'progress.json').write_text(json.dumps({'pid':os.getpid(),'arm':arm,'seed':seed,'epoch':epoch,'required_epochs':200,'completed_runs':len(reports),'required_runs':9,'checked_unix':time.time()}))
    if epoch%50==0:print(json.dumps({'arm':arm,'seed':seed,'epoch':epoch,'dev':metrics,'best_epoch':best_epoch}),flush=True)
   best_metrics=history[best_epoch]['dev'];report={'arm':arm,'seed':seed,'best_epoch':best_epoch,'best_dev':best_metrics,'parameters':sum(p.numel() for p in model.parameters())};retain(directory/'history.json',history);retain(directory/'summary.json',report);reports.append(report)
 retain(out/'summary.json',reports);(out/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed_runs':len(reports),'required_runs':9,'checked_unix':time.time()}));print(json.dumps({'completed_runs':len(reports),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}))
if __name__=='__main__':main()
