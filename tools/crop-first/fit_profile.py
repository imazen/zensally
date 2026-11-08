#!/usr/bin/env python3
"""Profile the real TRAIN loader/gradient path; no quality-selected checkpoint."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch.utils.data import DataLoader
from crop_fit import CropDataset,TrainingModel,MODEL_ARGS,loss,ResidentBatches

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--batch',type=int,default=64);p.add_argument('--threads',type=int,default=4);p.add_argument('--steps',type=int,default=200);p.add_argument('--warmup',type=int,default=20);p.add_argument('--resident-cache-mib',type=int,default=0);p.add_argument('--compile',action='store_true');p.add_argument('--nsys',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.manual_seed(31)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
    dataset=CropDataset(a.manifest,'fit',training=True)
    loader=ResidentBatches(dataset,a.batch,a.device,a.resident_cache_mib*1024*1024) if a.resident_cache_mib else DataLoader(dataset,batch_size=a.batch,num_workers=2,prefetch_factor=2,persistent_workers=True,pin_memory=a.device=='cuda',shuffle=False,generator=torch.Generator().manual_seed(31))
    iterator=iter(loader)
    base=TrainingModel().to(a.device,memory_format=torch.channels_last);model=torch.compile(base) if a.compile else base;model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,fused=a.device=='cuda');last=None
    events=[torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)] if a.device=='cuda' else None
    for step in range(a.warmup+a.steps):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(a.device,non_blocking=True) for k,v in batch.items()};batch['image']=batch['image'].contiguous(memory_format=torch.channels_last)
        if step==a.warmup and a.nsys:torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
        if step==a.warmup and events:events[0].record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):raw=model(**{k:batch[k] for k in MODEL_ARGS});value,_,_=loss(raw,batch)
        if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite real-data profiling loss')
        value.backward();torch.nn.utils.clip_grad_norm_(base.parameters(),5.,error_if_nonfinite=True);optimizer.step();last=float(value.detach())
    if events:events[1].record()
    if a.device=='cuda':torch.cuda.synchronize()
    if a.nsys:torch.cuda.cudart().cudaProfilerStop()
    result={'diagnostic_only':True,'real_train_sources':len(dataset),'canonical_holdouts_used':False,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'batch':a.batch,'warmup_steps':a.warmup,'profiled_steps':a.steps,'device':a.device,'final_loss':last,'resident_cache_limit_mib':a.resident_cache_mib,'resident_payload_bytes':getattr(loader,'payload_bytes',None),'cuda_peak_allocated':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda' else None,'checkpoints_written':False,'scope':'loader, transfer, forward, loss, backward and AdamW (fused on CUDA); excludes per-epoch dev evaluation/checkpoint I/O'}
    if events:result.update(step_ms=events[0].elapsed_time(events[1])/a.steps,all_gradients_finite=True,instrumented=a.nsys)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if events:print('RESULT '+json.dumps(result))
if __name__=='__main__':main()
