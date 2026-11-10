#!/usr/bin/env python3
"""Small shared-GPU operator checks, with no timing or artistic-quality claim."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from crop_regions import area_pool,area_pool_matmul

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('this caller explicitly requires CUDA')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.10);torch.manual_seed(29)
    original=torch.randn(2,8,8,12,dtype=torch.float64)
    rectangles=torch.tensor([[[0,0,1,1],[.13,.21,.79,.81],[.5,0,.501,1],[0,0,0,1]]]*2,dtype=torch.float64)
    rows=[]
    for name,dtype in [('fp64',torch.float64),('fp32',torch.float32),('bf16',torch.bfloat16),('fp16',torch.float16)]:
        quantized=original.to(dtype).to(torch.float64).requires_grad_()
        gold=area_pool(quantized,rectangles);gold_grad=torch.autograd.grad(gold.square().sum(),quantized)[0];gold=gold.detach()
        for backend,pool in [('integral',area_pool),('matmul',area_pool_matmul)]:
            image=original.to(device='cuda',dtype=dtype).requires_grad_()
            boxes=rectangles.to(device='cuda',dtype=torch.float64 if dtype==torch.float64 else torch.float32)
            value=pool(image,boxes);gradient=torch.autograd.grad(value.square().sum(),image)[0]
            value=value.detach().double().cpu();gradient=gradient.detach().double().cpu()
            if not torch.isfinite(value).all() or not torch.isfinite(gradient).all():raise ValueError('nonfinite pooling values/gradients')
            if dtype==torch.float64:
                torch.testing.assert_close(value,gold,rtol=1e-10,atol=1e-10);torch.testing.assert_close(gradient,gold_grad,rtol=1e-10,atol=1e-10)
            rows.append({'precision':name,'backend':backend,'max_absolute_value_error':float((value-gold).abs().max()),'max_absolute_gradient_error':float((gradient-gold_grad).abs().max()),'finite':True})
    result={'scope':'pooling operator suitability only; not whole-model or crop equivalence','device':torch.cuda.get_device_name(),'torch':torch.__version__,'timing_measured':False,'quality_measured':False,'allocator_cap_fraction':.10,'rows':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
