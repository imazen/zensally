#!/usr/bin/env python3
"""Registered matched-head suitability fits on frozen TRAIN fit/dev descriptors."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch import nn
from crop_fit import loss
from composition_head import CompositionHead

ARMS={'baseline':0,'pose':16,'pose-gaze':24,'pose-gaze-scan':32,'permuted':32}

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def predictions(cache,indices,raw):
 results=[]
 for bi,index in enumerate(indices):
  s=cache['sources'][index];groups={}
  for ci,c in enumerate(s['candidates']):groups.setdefault(c['task_id'],[]).append(ci)
  for tid,group in groups.items():
   ci=max(group,key=lambda j:float(raw[bi,j,0]));c=s['candidates'][ci];f=s['face_context'];box=c['box'];contains=f[0]==0 or (box[0]<=f[1] and box[1]<=f[2] and box[2]>=f[3] and box[3]>=f[4]);crop=bool(raw[bi,ci,1]>=0) and contains
   results.append({'source_id':s['id'],'family_id':s['family_id'],'task_id':tid,'ratio':c['ratio'],'mode':c['mode'],'raw_rank_candidate_id':c['candidate_id'],'candidate_id':c['candidate_id'] if crop else None,'xywh':c['xywh'],'research_action':'crop' if crop else 'contain','rank_teacher_acceptable':c['acceptable'],'judged':c['weight']>0,'acceptance_logit':float(raw[bi,ci,1]),'face_coverage_certified':False})
 return results

@torch.no_grad()
def evaluate(model,cache,indices,data,extra):
 raw=model(data['descriptors'][indices],extra[indices]);batch={k:v[indices] for k,v in data['packed'].items()};value,accept,rank=loss(raw,batch)
 return {'loss':float(value),'accept_loss':float(accept),'rank_loss':float(rank)},raw.cpu()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS));p.add_argument('--seeds',nargs='+',type=int,default=[31,137,911]);p.add_argument('--epochs',type=int,default=60);a=p.parse_args()
 if a.epochs!=60 or a.seeds!=[31,137,911]:raise ValueError('registered epochs and seeds must remain fixed')
 out=a.root/'head-fits';out.mkdir(exist_ok=False);torch.set_num_threads(2)
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 path=a.root/'feature-cache/features.pt';cache=torch.load(path,map_location='cpu',weights_only=False);identity=sha(path);fit=[i for i,u in enumerate(cache['uses']) if u=='fit'];dev=[i for i,u in enumerate(cache['uses']) if u=='dev']
 if (len(fit),len(dev))!=(64,16) or len(set(cache['ids']))!=80:raise ValueError('registered fitting roles changed')
 data={'descriptors':cache['descriptors'].to(a.device),'packed':{k:v.to(a.device) for k,v in cache['packed'].items()}};extra=cache['extras'].to(a.device);scorer=nn.Sequential(nn.Linear(data['descriptors'].shape[-1],96),nn.ReLU(),nn.Linear(96,2));scorer.load_state_dict(cache['scorer_state']);scorer.to(a.device);report=[]
 permutation=list(range(80));rng=torch.Generator().manual_seed(31)
 for indices in [fit,dev]:
  order=torch.randperm(len(indices),generator=rng).tolist()
  for i,j in enumerate(order):permutation[indices[i]]=indices[j]
 config={'arms':a.arms,'seeds':a.seeds,'epochs':60,'batch':16,'lr':.0001,'weight_decay':.01,'clip':5.,'cache_sha256':identity,'source_permutation':permutation,'canonical_holdouts_used':False,'selection':'minimum TRAIN-development loss per arm/seed; report all seeds; no threshold tuning','feature_input':'source-frame proxies; not artistic crop judgments','gradient_source_roles':['fit']}
 (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
 for arm in a.arms:
  for seed in a.seeds:
   torch.manual_seed(seed);model=CompositionHead(scorer,ARMS[arm]).to(a.device);features=extra[permutation] if arm=='permuted' else extra;directory=out/(arm+'-seed'+str(seed));directory.mkdir();optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01,fused=a.device=='cuda');history=[];best=float('inf');best_epoch=None
   with torch.no_grad():
    for i in range(80):
     actual=model(data['descriptors'][i:i+1],features[i:i+1]).cpu()
     if not torch.equal(actual,cache['initial_logits'][i:i+1]):raise ValueError('expanded head does not exactly reproduce frozen initializer')
   initial,_=evaluate(model,cache,dev,data,features);(directory/'initial.json').write_text(json.dumps(initial,indent=2)+'\n')
   for epoch in range(60):
    model.train();order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(seed+epoch)).tolist();total=0.
    for start in range(0,len(fit),16):
     indices=[fit[j] for j in order[start:start+16]];batch={k:v[indices] for k,v in data['packed'].items()};optimizer.zero_grad(set_to_none=True);raw=model(data['descriptors'][indices],features[indices]);value,_,_=loss(raw,batch)
     if not bool(torch.isfinite(value)):raise ValueError('nonfinite fitting loss')
     value.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();total+=float(value.detach())
    model.eval();metrics,raw=evaluate(model,cache,dev,data,features);row={'epoch':epoch+1,'fit_loss':total/4,'dev':metrics};history.append(row);improved=metrics['loss']<best
    if improved:best=metrics['loss'];best_epoch=epoch+1
    if improved or (epoch+1)%20==0:
     payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'arm':arm,'seed':seed,'cache_sha256':identity,'history':history,'protocol':config};target=directory/f'epoch-{epoch+1:03}.pt';torch.save(payload,target)
     if improved:
      alias=directory/'best.pt';stage=directory/'best.pt.new';stage.symlink_to(target.name);stage.replace(alias)
      prediction_path=directory/f'predictions-epoch-{epoch+1:03}.json';prediction_path.write_text(json.dumps(predictions(cache,dev,raw),indent=2)+'\n');prediction_link=directory/'best-predictions.json.new';prediction_link.symlink_to(prediction_path.name);prediction_link.replace(directory/'best-predictions.json')
    (out/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':'matched-head fitting','arm':arm,'seed':seed,'epoch':epoch+1,'required_epochs':60,'completed_runs':len(report),'required_runs':len(a.arms)*len(a.seeds),'last_checked_unix':time.time()}))
    if (epoch+1)%20==0:print(json.dumps({'arm':arm,'seed':seed,**row,'best_epoch':best_epoch}),flush=True)
   (directory/'history.json').write_text(json.dumps(history,indent=2)+'\n');result={'arm':arm,'seed':seed,'best_epoch':best_epoch,'best_dev_loss':best,'initial_dev':initial,'parameters':sum(p.numel() for p in model.parameters())};report.append(result);(directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(result),flush=True)
 if a.device=='cuda':torch.cuda.synchronize()
 (out/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','completed_runs':len(report),'required_runs':len(a.arms)*len(a.seeds),'last_checked_unix':time.time()}))
 print(json.dumps({'completed_runs':len(report),'required_runs':len(a.arms)*len(a.seeds),'cuda_peak_allocated':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda' else None}))
if __name__=='__main__':main()
