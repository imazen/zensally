#!/usr/bin/env python3
"""Real-image weak fitting with bounded loading, internal dev and resumable state."""
import argparse,hashlib,json,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from crop_fit import CropDataset,TrainingModel,MODEL_ARGS,loss

def transfer(batch,device):return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
class EpochSampler(Sampler):
    def __init__(self,count,seed):self.count=count;self.seed=seed;self.epoch=0
    def __iter__(self):return iter(torch.randperm(self.count,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist())
    def __len__(self):return self.count

def loader(dataset,a,epoch=0,shuffle=False):
    generator=torch.Generator().manual_seed(a.seed+epoch)
    kw={'num_workers':a.workers,'pin_memory':a.device=='cuda','generator':generator,'batch_size':a.batch,'drop_last':False}
    if shuffle:kw['sampler']=EpochSampler(len(dataset),a.seed)
    if a.workers:kw.update(prefetch_factor=2,persistent_workers=True)
    return DataLoader(dataset,**kw)

@torch.no_grad()
def evaluate(model,dataset,a,epoch=0,data_loader=None):
    model.eval();sums=[0.,0.,0.];batches=0;counts={'essential':{'tp':0,'tn':0,'positive':0,'negative':0,'tasks':0,'rank_acceptable':0,'rank_judged':0,'center_acceptable':0,'center_judged':0},'secondary':{'tp':0,'tn':0,'positive':0,'negative':0,'tasks':0,'rank_acceptable':0,'rank_judged':0,'center_acceptable':0,'center_judged':0}};predictions=[]
    for batch in (data_loader if data_loader is not None else loader(dataset,a)):
        batch=transfer(batch,a.device)
        with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.precision=='bf16'):raw=model(**{k:batch[k] for k in MODEL_ARGS})
        value,accept,rank=loss(raw,batch);sums=[x+float(y) for x,y in zip(sums,[value,accept,rank])];batches+=1
        values=raw.float().cpu();indices=batch['index'].cpu().tolist()
        for bi,source_index in enumerate(indices):
            source=dataset.sources[source_index];groups={}
            for ci,c in enumerate(source['candidates']):groups.setdefault(c['task_id'],[]).append(ci)
            for tid,group in groups.items():
                first=source['candidates'][group[0]];tier='essential' if 1/3<=first['aspect']<=3 else 'secondary';out=counts[tier];out['tasks']+=1
                for ci in group:
                    c=source['candidates'][ci]
                    if c['weight']<=0:continue
                    y=c['acceptable'];prediction=bool(values[bi,ci,1]>=0)
                    out['positive' if y else 'negative']+=1;out['tp' if y else 'tn']+=int(prediction==y)
                # These are unguarded research rankings, not deployable crops.
                chosen=max(group,key=lambda ci:float(values[bi,ci,0]));center=min(group,key=lambda ci:sum(((source['candidates'][ci]['box'][axis]+source['candidates'][ci]['box'][axis+2])/2-.5)**2 for axis in [0,1]))
                for name,ci in [('rank',chosen),('center',center)]:
                    c=source['candidates'][ci]
                    if c['weight']>0:out[name+'_judged']+=1;out[name+'_acceptable']+=int(c['acceptable'])
                c=source['candidates'][chosen];predictions.append({'source_id':source['id'],'task_id':tid,'candidate_id':c['candidate_id'],'xywh':c['xywh'],'teacher_acceptable':c['acceptable'],'judged':c['weight']>0,'acceptance_logit':float(values[bi,chosen,1]),'deployable':False,'face_coverage_reviewed':False})
    for out in counts.values():
        out['balanced_acceptability_accuracy']=((out['tp']/out['positive'] if out['positive'] else 0)+(out['tn']/out['negative'] if out['negative'] else 0))/2 if out['positive'] and out['negative'] else None
    return {'loss':sums[0]/batches,'accept_loss':sums[1]/batches,'rank_loss':sums[2]/batches,'counts':counts,'sources':len(dataset)},predictions

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--precision',choices=['fp32','bf16'],default='bf16');p.add_argument('--compile',action='store_true');p.add_argument('--epochs',type=int,default=100);p.add_argument('--stop-after-epoch',type=int);p.add_argument('--batch',type=int,default=16);p.add_argument('--workers',type=int,default=2);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=31);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--resume',type=Path);p.add_argument('--encoder-checkpoint',type=Path);p.add_argument('--evaluate-only',choices=['dev','eval']);a=p.parse_args()
    if a.epochs<1 or a.batch<1 or not 0<=a.workers<=4:raise ValueError('invalid fitting bounds')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA explicitly requested but unavailable')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cudnn.benchmark=True
    a.output.mkdir(parents=True,exist_ok=False);identity=hashlib.sha256(a.manifest.read_bytes()).hexdigest();config={k:getattr(a,k) for k in ['device','precision','compile','batch','workers','threads','seed','lr','epochs']}
    fit_data=CropDataset(a.manifest,'fit',training=True);dev_data=CropDataset(a.manifest,a.evaluate_only or 'dev');base=TrainingModel().to(a.device,memory_format=torch.channels_last)
    initialization=None
    if a.encoder_checkpoint:
        state=torch.load(a.encoder_checkpoint,map_location='cpu',weights_only=True)
        encoder={k:v for k,v in state.items() if k.split('.')[0] in ['stem','enc1','enc2','enc3','bottleneck']}
        expected={k for k in base.ranker.state_dict() if k.split('.')[0] in ['stem','enc1','enc2','enc3','bottleneck']}
        if set(encoder)!=expected:raise ValueError('encoder checkpoint does not cover the complete owned encoder')
        base.ranker.load_state_dict(encoder,strict=False);initialization={'path':str(a.encoder_checkpoint),'sha256':hashlib.sha256(a.encoder_checkpoint.read_bytes()).hexdigest(),'tensors':len(encoder)}
    optimizer=torch.optim.AdamW(base.parameters(),lr=a.lr,weight_decay=.01,fused=a.device=='cuda');start_epoch=0;best=float('inf');best_epoch=None;history=[]
    if a.resume:
        state=torch.load(a.resume,map_location=a.device,weights_only=False)
        if state['manifest_sha256']!=identity or state['config']!=config:raise ValueError('resume changes frozen data or optimizer/run configuration')
        base.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_epoch=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];history=state['history'];initialization=state['initialization']
        torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if a.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    model=torch.compile(base) if a.compile else base
    if a.evaluate_only:
        if not a.resume:raise ValueError('evaluation requires an explicit fitted checkpoint')
        metrics,predictions=evaluate(model,dev_data,a);(a.output/'evaluation.json').write_text(json.dumps({'use':a.evaluate_only,'metrics':metrics,'manifest_sha256':identity,'human_rated':False,'predictions':predictions},indent=2)+'\n');print(json.dumps(metrics));return
    fit_loader=loader(fit_data,a,shuffle=True);dev_loader=loader(dev_data,a)
    started=time.monotonic();initial,_=evaluate(model,dev_data,a,data_loader=dev_loader);save_initial={'config':config,'manifest_sha256':identity,'fit_sources':len(fit_data),'dev_sources':len(dev_data),'initialization':initialization,'initial_dev':initial,'human_rated':False,'per_loader_worker_image_cache_bytes':64*1024*1024,'prefetch_batches_per_worker':2,'candidate_context_packed_once':True}
    (a.output/'run.json').write_text(json.dumps(save_initial,indent=2)+'\n')
    final_predictions=[]
    for epoch in range(start_epoch,a.epochs):
        model.train();total=0.;steps=0
        fit_loader.sampler.epoch=epoch
        for batch in fit_loader:
            batch=transfer(batch,a.device);batch['image']=batch['image'].contiguous(memory_format=torch.channels_last);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.precision=='bf16'):raw=model(**{k:batch[k] for k in MODEL_ARGS});value,_,_=loss(raw,batch)
            if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite real-data training loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(base.parameters(),5.,error_if_nonfinite=True);optimizer.step();total+=float(value.detach());steps+=1
        metrics,predictions=evaluate(model,dev_data,a,data_loader=dev_loader);row={'epoch':epoch+1,'train_loss':total/steps,'dev':metrics};history.append(row)
        improved=metrics['loss']<best
        if improved:best=metrics['loss'];best_epoch=epoch+1;final_predictions=predictions
        state={'model':base.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'history':history,'config':config,'manifest_sha256':identity,'initialization':initialization,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if a.device=='cuda' else None,'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        target=a.output/'last.pth';staged=a.output/'last.writing.pth';torch.save(state,staged);staged.replace(target)
        if improved:torch.save(state,a.output/'best.pth');(a.output/'best-predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
        (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');(a.output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':'fitting','epoch':epoch+1,'required_epochs':a.epochs,'best_epoch':best_epoch,'best_dev_loss':best,'checked_at':datetime.now(timezone.utc).isoformat()}))
        print(json.dumps({'epoch':epoch+1,'train_loss':row['train_loss'],'dev_loss':metrics['loss'],'best_epoch':best_epoch}),flush=True)
        if a.stop_after_epoch and epoch+1>=a.stop_after_epoch:break
    if a.device=='cuda':torch.cuda.synchronize()
    result={'completed_epochs':history[-1]['epoch'],'required_epochs':a.epochs,'best_epoch':best_epoch,'best_dev_loss':best,'elapsed_seconds':time.monotonic()-started,'cuda_peak_allocated':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda' else None,'manifest_sha256':identity,'human_rated':False,'canonical_holdouts_used_for_selection':False,'deployable':False}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
