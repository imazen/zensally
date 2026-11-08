#!/usr/bin/env python3
"""Bounded synthetic training-path probe. No admitted artistic quality claims."""
import argparse,atexit,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch import nn
from torch.nn import functional as F
from crop_ranker import CropRanker
from crop_regions import RegionCropRanker


def fixture(batch,count,height,width,device,seed):
    g=torch.Generator(device=device).manual_seed(seed)
    image=torch.rand(batch,3,height,width,generator=g,device=device)
    lo=torch.rand(batch,count,2,generator=g,device=device)*.5
    hi=(lo+.25+torch.rand(batch,count,2,generator=g,device=device)*.5).clamp_max(1)
    boxes=torch.cat((lo,hi),-1)
    # Exact synthetic central-subject overlap supplies diagnostic labels only.
    intersect=(torch.minimum(hi,torch.full_like(hi,.65))-torch.maximum(lo,torch.full_like(lo,.35))).clamp_min(0).prod(-1)/.09
    targets=(intersect[:,::2]>intersect[:,1::2]).float()
    targets=torch.where(intersect[:,::2]==intersect[:,1::2],.5,targets)
    acceptable=(intersect>=.8).float()
    args=(boxes,torch.full((batch,),width/height,device=device),((hi[...,0]-lo[...,0])/(hi[...,1]-lo[...,1])),torch.ones(batch,count,device=device),torch.tensor([[0,0,0,0,0,1]],device=device).expand(batch,-1),torch.ones(batch,count,dtype=torch.bool,device=device),torch.tensor([[0,0,1,1]],device=device).expand(batch,-1))
    return image,args,targets,acceptable


class Objective(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,image,args,targets,acceptable):
        output=self.model(image,*args)
        rank,accept=output if isinstance(output,tuple) else (output,None)
        loss=F.binary_cross_entropy_with_logits((rank[:,::2]-rank[:,1::2]).float(),targets)
        if accept is not None:loss=loss+F.binary_cross_entropy_with_logits(accept.float(),acceptable)
        return loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--model',choices=['point','regions'],default='regions');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--candidates',type=int,default=64);p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=256)
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='fp32');p.add_argument('--channels-last',action='store_true');p.add_argument('--compile',choices=['none','default','reduce-overhead'],default='none')
    p.add_argument('--optimizer',choices=['loop','foreach','fused'],default='foreach');p.add_argument('--threads',type=int,default=4);p.add_argument('--warmup',type=int,default=10);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--profile',action='store_true');p.add_argument('--graph',action='store_true');p.add_argument('--upload',action='store_true');p.add_argument('--nsys',action='store_true');a=p.parse_args()
    if not 1<=a.batch<=256 or not 2<=a.candidates<=512 or a.candidates%2 or min(a.height,a.width)<32 or max(a.height,a.width)>512 or not 1<=a.steps<=20000 or not 1<=a.warmup<=1000:raise ValueError('bounded batch/candidate/input/work count required')
    a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(23);torch.set_num_threads(a.threads)
    device=torch.device(a.device);cuda=a.device=='cuda'
    if cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_per_process_memory_fraction(.70)
        torch.backends.cudnn.benchmark=True
        telemetry=(a.output/'gpu.csv').open('x')
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits','--loop-ms=200'],stdout=telemetry,stderr=subprocess.STDOUT)
        def stop_monitor():
            if monitor.poll() is None:monitor.terminate()
            monitor.wait(timeout=10);telemetry.close()
        atexit.register(stop_monitor)
    torch.set_float32_matmul_precision('highest')
    model=(CropRanker() if a.model=='point' else RegionCropRanker()).to(device).train()
    image,args,targets,acceptable=fixture(a.batch,a.candidates,a.height,a.width,device,23)
    if a.channels_last:model=model.to(memory_format=torch.channels_last);image=image.contiguous(memory_format=torch.channels_last)
    pinned=None
    if a.upload:
        if not cuda:raise ValueError('upload probe needs CUDA')
        pinned=(image*255).to(torch.uint8).cpu().pin_memory()
    objective=Objective(model)
    if a.compile!='none':objective=torch.compile(objective,mode=a.compile,fullgraph=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=a.optimizer=='foreach',fused=a.optimizer=='fused',capturable=a.graph)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[a.precision]
    if a.graph and a.precision=='fp16':raise ValueError('FP16 scaling is not graph-captured by this probe')
    scaler=torch.amp.GradScaler('cuda',enabled=cuda and a.precision=='fp16')
    initial={n:v.detach().clone() for n,v in model.named_parameters()}
    def step():
        nonlocal image
        if pinned is not None:image=pinned.to(device,non_blocking=True).float().div_(255)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=a.device,dtype=dtype,enabled=a.precision!='fp32'):
            loss=objective(image,args,targets,acceptable)
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        return loss
    status={'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'stage':'warmup','config':vars(a)|{'output':str(a.output)},'torch':torch.__version__,'cuda':torch.version.cuda}
    def record(stage):status['stage']=stage;status['checked_at']=datetime.now(timezone.utc).isoformat();(a.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    record('warmup')
    if a.graph:
        if not cuda or a.upload:raise ValueError('graph probe needs resident CUDA fixtures')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(a.warmup):loss=step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True);del loss
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):loss=step()
        graph.replay();torch.cuda.synchronize()
        def execute():graph.replay();return loss
    else:
        for _ in range(a.warmup):loss=step()
        execute=step
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    initial_loss=float(loss.detach());record('measuring')
    if a.nsys:
        if not cuda:raise ValueError('CUDA capture requested for CPU run')
        torch.cuda.profiler.start()
    events=[];memory=[];start=time.perf_counter()
    for i in range(a.steps):
        if cuda:
            before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True);before.record()
        loss=execute()
        if cuda:after.record();events.append((before,after))
        if i%max(1,a.steps//5)==0 and cuda:memory.append({'step':i,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()})
    if cuda:torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    if a.nsys:torch.cuda.profiler.stop()
    final_loss=float(loss.detach())
    gradients_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    changed=sum(not torch.equal(v,initial[n]) for n,v in model.named_parameters())
    if not gradients_finite or not changed or not torch.isfinite(loss):raise RuntimeError('training suitability failed: nonfinite/no update')
    peak_allocated=torch.cuda.max_memory_allocated() if cuda else None
    peak_reserved=torch.cuda.max_memory_reserved() if cuda else None
    if a.profile:
        record('profiling')
        activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
            for _ in range(3):execute()
            if cuda:torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.output/'trace.json'))
        (a.output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total' if cuda else 'self_cpu_time_total',row_limit=35))
    result={**status,'stage':'complete','synthetic_only':True,'quality_measured':False,'artistic_labels_used':False,'device_name':torch.cuda.get_device_name() if cuda else 'CPU','parameters':sum(v.numel() for v in model.parameters()),'elapsed_seconds':elapsed,'step_ms':elapsed/a.steps*1000,'images_per_second':a.batch*a.steps/elapsed,'cuda_event_ms':[b.elapsed_time(e) for b,e in events],'initial_postwarmup_loss':initial_loss,'final_loss':final_loss,'all_gradients_finite':gradients_finite,'updated_parameter_tensors':changed,'cuda_peak_allocated':peak_allocated,'cuda_peak_reserved':peak_reserved,'memory_samples':memory,'source_sha256':{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*(Path(__file__).resolve().parents[2]/'training'/n for n in ['crop_regions.py','crop_ranker.py','model_v3ds.py'])]}}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');record('complete')
    print('RESULT '+json.dumps({k:result[k] for k in ['step_ms','images_per_second','initial_postwarmup_loss','final_loss','cuda_peak_allocated','all_gradients_finite']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=Path(sys.argv[1])/'progress.json'
        if path.exists():
            state=json.loads(path.read_text());state.update(stage='failed',error=str(error),checked_at=datetime.now(timezone.utc).isoformat());path.write_text(json.dumps(state,indent=2)+'\n')
        raise
