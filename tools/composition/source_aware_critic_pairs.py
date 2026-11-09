#!/usr/bin/env python3
"""Bounded, source-aware and order-swapped local critic suitability diagnostic."""
import argparse
import atexit
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from critic_probe import sha

PREFIX = ('Compare two candidate crops of the same source photograph. The first image is '
          'the complete source; the second is crop A; the third is crop B. ')
SUFFIX = ('Choose the stronger artistic composition for the stated crop mode, preserving '
          'a coherent subject or story. Prefer intact visible faces; excluding a person '
          'entirely is allowed when the ratio cannot keep everyone. Minimal means keeping '
          'more context; Maximal means a tighter framing. Do not judge different modes '
          'against one another. Return only JSON with keys "choice" ("A", "B", or "tie") '
          'and "reason_code" ("subject", "face", "balance", "context", or "similar").')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'processor', 'plan', 'gallery', 'resizer', 'output']:
        p.add_argument(key, type=Path)
    p.add_argument('--limit', type=int, required=True, choices=range(1,13))
    a = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cuda.matmul.allow_tf32 = False
    if sha(a.model/'asset-identities.json') != 'aab4ad8df6e3e4bb45e0e69f10e1a0c1ac0c723ff7704ca05e6cd49ace26c570' or sha(a.plan) != '51b788924c962a93ac9e8ec069ec72a147bf08baa06e7fdb5342e2153cc5b901' or sha(a.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('registered model/plan/resizer differs')
    for row in json.loads((a.model/'asset-identities.json').read_text())['files']:
        path = a.model/row['name']
        if not path.resolve().is_relative_to(a.model.resolve()) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('model member differs')
    plan = json.loads(a.plan.read_text())
    gallery = json.loads((a.gallery/'summary.json').read_text())
    if len(gallery['cases']) != 6 or gallery['plan_sha256'] != sha(a.plan):
        raise ValueError('six frozen pairs required')
    original = json.loads((a.model/'config.json').read_text())
    if original.get('use_cache') is not None:
        raise ValueError('unexpected original loader configuration')
    original['use_cache'] = True
    if json.loads((a.processor/'config.json').read_text()) != original:
        raise ValueError('derived processor configuration differs')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(plan_sha256=sha(a.plan), gallery_sha256=sha(a.gallery/'summary.json'),
                    assets_sha256=sha(a.model/'asset-identities.json'), prefix=PREFIX, suffix=SUFFIX,
                    max_new_tokens=256, seed=31, precision='BF16', weight_placement='GPU10GiB CPU10GiB; no disk',
                    source_max_axis=336, source_grid_multiple=28, requested_calls=12,
                    api_calls=0, admit_training_labels=False, test_inference=False)
    dest = a.output/'protocol.json'
    if dest.exists():
        if json.loads(dest.read_text()) != protocol:
            raise ValueError('cached protocol differs')
    else:
        write(dest, protocol)
    status = dict(pid=os.getpid(), stage='prepare-inputs', completed=0, required=a.limit)
    def progress(**kw):
        status.update(kw, checked_unix=time.time())
        (a.output/'PROGRESS.json').write_text(json.dumps(status, indent=2)+'\n')
    def failed():
        if status['pid'] is not None:
            progress(pid=None, stage='failed-artifacts-retained')
    atexit.register(failed)
    progress()
    tasks = []
    for index, case in enumerate(gallery['cases']):
        rows = [r for r in plan['images'] if r.get('task_id') == case['task_id']]
        if len(rows) != 2 or {r['weak_teacher_acceptable'] for r in rows} != {True,False}:
            raise ValueError('fixed weak pair differs')
        accepted = next(r for r in rows if r['weak_teacher_acceptable'])
        rejected = next(r for r in rows if not r['weak_teacher_acceptable'])
        pane = case['panes'][0]
        source = a.gallery/f'case{index}-source.rgb'
        if sha(source) != pane['rgb_sha256']:
            raise ValueError('owned source context differs')
        w,h = pane['width'],pane['height']
        factor = min(1,336/max(w,h))
        dw,dh = [max(28,int(round(n*factor/28))*28) for n in (w,h)]
        path = a.output/f'case{index}-source.rgb'
        raw = subprocess.check_output([str(a.resizer),str(w),str(h),str(w*3),f'0,0,{w},{h},{dw},{dh}'],input=source.read_bytes())
        if len(raw) != dw*dh*3:
            raise ValueError('owned preview length differs')
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('cached preview differs')
        else:
            with path.open('xb') as f:f.write(raw)
        src = dict(rgb=str(path),width=dw,height=dh,sha256=sha(path),source_aspect=w/h,preview_aspect=dw/dh)
        parts = case['task_id'].split('-')
        prompt = PREFIX+f'Target aspect ratio: {parts[1].replace("x",":")}. Mode: {parts[2]}. '+SUFFIX
        for order,pair in [('AB',[accepted,rejected]),('BA',[rejected,accepted])]:
            images = [src,*[{k:r[k] for k in ['rgb','width','height','sha256']} for r in pair]]
            tasks.append(dict(id=case['task_id']+'-'+order, task=case['task_id'],images=images,prompt=prompt,
                              expected='A' if order=='AB' else 'B',order=order))
    processor = AutoProcessor.from_pretrained(a.processor,local_files_only=True)
    progress(stage='load-model')
    # Same qualified BF16 loading contract as critic_probe; source config is explicit.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,config=Qwen2_5_VLConfig.from_dict(original),local_files_only=True,dtype=torch.bfloat16,device_map='auto',max_memory={0:'10GiB','cpu':'10GiB'},attn_implementation='sdpa').eval()
    if any(type(v).__name__=='Params4bit' for v in model.parameters()) or 'disk' in model.hf_device_map.values():
        raise ValueError('unquantized/no-disk placement differs')
    placement = dict(modules=model.hf_device_map,execution_devices={name:str(m._hf_hook.execution_device) for name,m in model.named_modules() if hasattr(m,'_hf_hook') and hasattr(m._hf_hook,'execution_device')},parameter_dtypes=sorted({str(v.dtype) for v in model.parameters()}))
    dest = a.output/'placement.json'
    if dest.exists():
        if json.loads(dest.read_text()) != placement:raise ValueError('placement changed')
    else:write(dest,placement)
    torch.manual_seed(31)
    results=[]
    for task in tasks[:a.limit]:
        dest = a.output/(task['id']+'.json')
        if dest.exists():
            prior=json.loads(dest.read_text())
            if prior['input'] != task:raise ValueError('cached task differs')
            results.append(prior)
            continue
        arrays=[]
        for row in task['images']:
            path=Path(row['rgb']);w,h=row['width'],row['height']
            if w%28 or h%28 or w*h>448**2 or sha(path)!=row['sha256']:raise ValueError('image identity/grid bound differs')
            arrays.append(np.frombuffer(path.read_bytes(),dtype=np.uint8).reshape(h,w,3).copy())
        content=[{'type':'image'} for _ in arrays]+[{'type':'text','text':task['prompt']}]
        text=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=arrays,do_resize=False,return_tensors='pt').to('cuda')
        grid=inputs['image_grid_thw'].cpu().tolist()
        if grid != [[1,r['height']//14,r['width']//14] for r in task['images']]:raise ValueError('processor altered owned dimensions')
        progress(stage='source-aware-pair-inference',id=task['id'],completed=len(results))
        with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True)
        generated=tokens[:,inputs.input_ids.shape[1]:]
        response=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        choice=None;error=None;reason=None
        try:
            value=json.loads(response)
            if set(value)!={'choice','reason_code'} or value['choice'] not in ['A','B','tie'] or value['reason_code'] not in ['subject','face','balance','context','similar']:raise ValueError('invalid choice schema')
            choice=value['choice'];reason=value['reason_code']
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        result=dict(input=task,response=response,choice=choice,reason_code=reason,parse_error=error,
                    input_tokens=inputs.input_ids.shape[1],output_tokens=generated.shape[1],image_grid_thw=grid,
                    token_limit_reached=generated.shape[1]==256,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
        write(dest,result);results.append(result);progress(completed=len(results))
        print(json.dumps(dict(id=task['id'],choice=choice,parse_error=error,tokens=generated.shape[1])),flush=True)
    parsed=sum(r['choice'] is not None for r in results);correct=sum(r['choice']==r['input']['expected'] for r in results)
    pairs=[]
    for index in range(0,len(results)-1,2):
        x,y=results[index:index+2];xc,yc=x['choice'],y['choice']
        consistent=xc is not None and yc is not None and ((xc==yc=='tie') or (xc in ['A','B'] and yc in ['A','B'] and xc!=yc))
        pairs.append(dict(task=x['input']['task'],choices=[xc,yc],swap_consistent=consistent,both_correct=xc=='A' and yc=='B'))
    summary=dict(required=a.limit,completed=len(results),parsed=parsed,correct=correct,pairs=pairs,
                 truncated=sum(r['token_limit_reached'] for r in results),qualification_passed=len(results)==12 and parsed==12 and correct==12 and not any(r['token_limit_reached'] for r in results) and all(r['swap_consistent'] for r in pairs),
                 api_calls=0,admit_training_labels=False,test_inference=False,model_promoted=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress(pid=None,stage='complete',completed=len(results))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
