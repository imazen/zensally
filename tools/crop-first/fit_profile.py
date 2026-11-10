#!/usr/bin/env python3
"""Profile the real TRAIN loader/gradient path; no quality-selected checkpoint."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch.utils.data import DataLoader
from crop_fit import CropDataset,TrainingModel,MODEL_ARGS,loss,ResidentBatches

class AuxiliaryProfileDataset:
    """Repeated TRAIN-only geometry sources for explicitly sized profiling batches."""
    def __init__(self,path,repeats):
        data=torch.load(path,map_location='cpu',weights_only=False)
        self.indices=[i for i,s in enumerate(data['sources']) if s['use']=='fit']*repeats
        if not self.indices or repeats<1 or repeats>8:raise ValueError('bounded auxiliary TRAIN repeats required')
        self.data=data
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]
        return {'image':self.data['image'][i],**{k:v[i] for k,v in self.data['packed'].items()}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--batch',type=int,default=64);p.add_argument('--threads',type=int,default=4);p.add_argument('--steps',type=int,default=200);p.add_argument('--warmup',type=int,default=20);p.add_argument('--resident-cache-mib',type=int,default=0);p.add_argument('--compile',action='store_true');p.add_argument('--compile-mode',choices=['default','reduce-overhead'],default='default');p.add_argument('--nsys',action='store_true');p.add_argument('--freeze-batchnorm',action='store_true');p.add_argument('--disable-tf32',action='store_true');p.add_argument('--active-head',choices=['both','auxiliary'],default='both');p.add_argument('--auxiliary',action='store_true');p.add_argument('--source-repeats',type=int,default=1);p.add_argument('--precision',choices=['fp32','bf16'],default='bf16');p.add_argument('--pool-backend',choices=['integral','matmul'],default='integral');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.manual_seed(31)
    if a.disable_tf32:torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
    dataset=AuxiliaryProfileDataset(a.manifest,a.source_repeats) if a.auxiliary else CropDataset(a.manifest,'fit',training=True)
    loader=ResidentBatches(dataset,a.batch,a.device,a.resident_cache_mib*1024*1024) if a.resident_cache_mib else DataLoader(dataset,batch_size=a.batch,num_workers=2,prefetch_factor=2,persistent_workers=True,pin_memory=a.device=='cuda',shuffle=False,generator=torch.Generator().manual_seed(31))
    iterator=iter(loader)
    base=TrainingModel();base.ranker.pool_backend=a.pool_backend
    if a.auxiliary:
        from composition_joint import AuxiliaryScorer,auxiliary_loss
        base.ranker.scorer=AuxiliaryScorer(base.ranker.scorer);base.ranker.scorer.task=a.active_head
    base.to(a.device,memory_format=torch.channels_last);model=torch.compile(base,mode=a.compile_mode) if a.compile else base;model.train();
    if a.freeze_batchnorm:
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,fused=a.device=='cuda');last=None
    events=[torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)] if a.device=='cuda' else None
    for step in range(a.warmup+a.steps):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(a.device,non_blocking=True) for k,v in batch.items()};batch['image']=batch['image'].contiguous(memory_format=torch.channels_last)
        if step==a.warmup and a.nsys:torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
        if step==a.warmup and events:events[0].record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda' and a.precision=='bf16'):
            raw=model(**{k:batch[k] for k in MODEL_ARGS})
            if a.auxiliary:value=auxiliary_loss(raw,batch['targets'],batch['mask'])
            else:value,_,_=loss(raw,batch)
        if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite real-data profiling loss')
        value.backward();torch.nn.utils.clip_grad_norm_(base.parameters(),5.,error_if_nonfinite=True);optimizer.step();last=float(value.detach())
    if events:events[1].record()
    if a.device=='cuda':torch.cuda.synchronize()
    if a.nsys:torch.cuda.cudart().cudaProfilerStop()
    result={'diagnostic_only':True,'real_train_sources':len(dataset),'canonical_holdouts_used':False,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'batch':a.batch,'warmup_steps':a.warmup,'profiled_steps':a.steps,'device':a.device,'final_loss':last,'resident_cache_limit_mib':a.resident_cache_mib,'resident_payload_bytes':getattr(loader,'payload_bytes',None),'cuda_peak_allocated':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda' else None,'checkpoints_written':False,'scope':'loader, transfer, forward, loss, backward and AdamW (fused on CUDA); excludes per-epoch dev evaluation/checkpoint I/O'}
    if a.auxiliary:
        result.pop('real_train_sources');result.update(auxiliary_geometry=True,synthetic=True,profile_source_rows=len(dataset),unique_train_sources=len(set(dataset.indices)),profile_source_repeats=a.source_repeats)
    result.update(active_head=a.active_head,compiled=a.compile,compile_mode=a.compile_mode if a.compile else None,precision=a.precision,pool_backend=a.pool_backend,batchnorm_running_stats_frozen=a.freeze_batchnorm,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if events:result.update(step_ms=events[0].elapsed_time(events[1])/a.steps,all_gradients_finite=True,instrumented=a.nsys)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if events:print('RESULT '+json.dumps(result))
if __name__=='__main__':main()
