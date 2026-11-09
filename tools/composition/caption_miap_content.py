#!/usr/bin/env python3
"""Cache local vision captions as content-review hints; never training admission."""
import argparse
import atexit
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import Florence2ForConditionalGeneration,Florence2Processor

from artifact_identity import sha
from probe_florence_regions import retain

TASK='<MORE_DETAILED_CAPTION>'
DOCUMENT_PATTERNS=['magazine cover','cover of a magazine','newspaper page','page of a newspaper',
    'screenshot','book cover','cover of a book','scanned document','text document','poster with','poster for']
DEPICTION_PATTERNS=['statue','sculpture','mannequin','cardboard cutout','painting of','illustration of','drawing of']


def hints(text):
    lower=text.casefold()
    return dict(document=[p for p in DOCUMENT_PATTERNS if p in lower],depiction=[p for p in DEPICTION_PATTERNS if p in lower])


def sources(root,stage):
    if stage=='witnesses':
        if sha(root/'SHA256SUMS.json')!='8468972acdcd5edc34bf2f7042631d83bc104071cf2bd99e9147dd0f280316ad':raise ValueError('frozen witness packet differs')
        protocol=json.loads((root/'protocol.json').read_text())
        rows=[json.loads((root/identifier.split(':')[1]/'record.json').read_text()) for identifier in protocol['selected_ids']]
        if len(rows)!=12 or any(r['source']['role']!='fit' for r in rows):raise ValueError('twelve fitting witnesses required')
    else:
        summary=json.loads((root/'summary.json').read_text());manifest=json.loads((root/'manifest.json').read_text());rows=manifest['rows']
        if summary['sources']!=384 or summary['independent_png_readbacks']!=384 or Counter(r['source']['role'] for r in rows)!=Counter(fit=288,dev=96):raise ValueError('complete fit/development acquisition required')
        if manifest['protocol']['plan_packet_sha256']!='e5baf11ae1673af8b1feb98e790cbd17f93976f5c37ebe3888af3d508bd0a5db':raise ValueError('corrected source roles differ')
    if any(r['training_eligible'] or r['source']['role'] not in {'fit','dev'} for r in rows):raise ValueError('unadmitted fit/development sources only')
    return rows


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['assets','sources','resizer','output']:p.add_argument(name,type=Path)
    p.add_argument('--stage',choices=['witnesses','pilot'],required=True);a=p.parse_args()
    rows=sources(a.sources,a.stage)
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':raise ValueError('qualified owned resizer differs')
    assets=json.loads((a.assets/'asset-identities.json').read_text())
    if assets['revision']!='00921df66db728a9ceb750f5eca43e5c203a2051' or assets['remote_code_permitted']:raise ValueError('native local asset contract differs')
    for row in assets['files']:
        path=a.assets/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('local model asset differs')
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','responses']:(a.output/name).mkdir(exist_ok=True)
    if (a.output/(a.stage+'-summary.json')).exists():raise ValueError('caption stage already complete; use cached responses')
    status=dict(pid=os.getpid(),stage='loading-local-model',completed=0,required=len(rows),checked_unix=time.time())
    progress=a.output/'progress.json';progress.write_text(json.dumps(status)+'\n')
    def final_status():
        if status['pid']==os.getpid():
            status.update(pid=None,stage='failed-cached-responses-retained',checked_unix=time.time());progress.write_text(json.dumps(status)+'\n')
    atexit.register(final_status)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(31)
    torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,loading=Florence2ForConditionalGeneration.from_pretrained(a.assets/'model',dtype=torch.float16,
        attn_implementation='eager',local_files_only=True,trust_remote_code=False,use_safetensors=True,output_loading_info=True)
    retain(a.output/(a.stage+'-loading.json'),loading)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('native checkpoint initialization incomplete')
    model=model.eval().cuda();processor=Florence2Processor.from_pretrained(a.assets/'model',local_files_only=True,trust_remote_code=False)
    generation=dict(max_new_tokens=256,num_beams=1,do_sample=False,use_cache=True)
    identity=dict(repo=assets['repo'],revision=assets['revision'],asset_manifest_sha256=sha(a.assets/'asset-identities.json'),
        weight_sha256=next(r['sha256'] for r in assets['files'] if r['path']=='model/model.safetensors'),
        model_class=type(model).__name__,processor_class=type(processor).__name__,dtype='float16',attention='eager',
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','huggingface-hub']},
        model_parameters=sum(p.numel() for p in model.parameters()),generation_config=model.generation_config.to_dict(),generation=generation,
        preprocessing='owned zenresize Lanczos stretch768; processor spatial transforms disabled',resizer_sha256=sha(a.resizer),
        image_mean=processor.image_processor.image_mean,image_std=processor.image_processor.image_std,remote_code_executed=False)
    identity=json.loads(json.dumps(identity));retain(a.output/'model-identity.json',identity)
    results=[]
    for row in rows:
        key=row['id'].split(':')[1];rgb_path=a.sources/key/'preview.rgb';width,height=row['geometry']['width'],row['geometry']['height']
        if sha(rgb_path)!=row['preview.rgb_sha256']:raise ValueError('owned input pixels differ')
        status.update(stage='caption-content',current_id=row['id'],checked_unix=time.time());progress.write_text(json.dumps(status)+'\n')
        path=a.output/'inputs'/(key+'.rgb');meta=path.with_suffix('.json')
        input_contract=dict(id=row['id'],source_rgb_sha256=row['preview.rgb_sha256'],working_size=[width,height],output_size=[768,768],resizer_sha256=sha(a.resizer))
        if meta.exists():
            saved=json.loads(meta.read_text())
            if saved['contract']!=input_contract or sha(path)!=saved['rgb_sha256']:raise ValueError('cached model input differs')
        else:
            with rgb_path.open('rb') as stream:rgb=subprocess.check_output([str(a.resizer),str(width),str(height),str(width*3),f'0,0,{width},{height},768,768'],stdin=stream)
            if len(rgb)!=768*768*3:raise ValueError('owned model-input extent differs')
            with path.open('xb') as stream:stream.write(rgb)
            retain(meta,dict(contract=input_contract,rgb_sha256=sha(path)))
        contract=dict(identity=identity,source_id=row['id'],source_role=row['source']['role'],input_rgb_sha256=sha(path),task=TASK,prompt=TASK)
        contract=json.loads(json.dumps(contract));digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest();response=a.output/'responses'/(digest+'.json')
        reused=response.exists()
        if reused:
            result=json.loads(response.read_text())
            if result['contract']!=contract or result['contract_sha256']!=digest:raise ValueError('cached content response differs')
        else:
            image=np.frombuffer(path.read_bytes(),np.uint8).reshape(768,768,3).copy()
            inputs=processor(text=TASK,images=image,return_tensors='pt',do_resize=False,do_center_crop=False)
            expected=image.astype(np.float32).transpose(2,0,1)/255
            expected=(expected-np.array(identity['image_mean'],np.float32)[:,None,None])/np.array(identity['image_std'],np.float32)[:,None,None]
            actual=inputs['pixel_values'].float().numpy()
            if actual.shape!=(1,3,768,768) or not np.isfinite(actual).all():raise ValueError('unexpected normalized input')
            error=float(np.abs(actual[0]-expected).max())
            if error>1e-6:raise ValueError('processor changed spatial inputs beyond pointwise normalization')
            device={k:v.to(device='cuda',dtype=torch.float16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            generated=model.generate(**device,**generation);tokens=generated[0].cpu().tolist();raw=processor.batch_decode(generated,skip_special_tokens=False)[0]
            eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            result=dict(contract=contract,contract_sha256=digest,token_ids=tokens,generated_text=raw,
                token_cap_reached=len(tokens)-1>=generation['max_new_tokens'],ended_with_eos=bool(tokens and tokens[-1] in eos),
                normalization_max_absolute_error=error,training_eligible=False,face_coverage_certified=False)
            try:result['parsed']=processor.post_process_generation(text=raw,task=TASK,image_size=tuple(row['header']['size']));result['postprocess_error']=None
            except Exception as err:result['parsed']=None;result['postprocess_error']=repr(err)
            retain(response,result)
        invalid=result['token_cap_reached'] or not result['ended_with_eos'] or result['postprocess_error'] is not None
        text=result['parsed'].get(TASK,'') if isinstance(result['parsed'],dict) else ''
        results.append(dict(id=row['id'],role=row['source']['role'],response=response.name,reused=reused,invalid=invalid,caption=text,review_hints=hints(text)))
        status.update(completed=len(results),checked_unix=time.time());progress.write_text(json.dumps(status)+'\n')
        if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>1024**3:raise ValueError('oneGiB content-cache allowance exceeded; retain all bytes')
        print(json.dumps(dict(completed=len(results),required=len(rows),**results[-1])),flush=True)
    summary=dict(stage=a.stage,completed=len(results),required=len(rows),invalid_calls=sum(r['invalid'] for r in results),cached_calls=sum(r['reused'] for r in results),calls=results,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        document_hint_sources=sum(bool(r['review_hints']['document']) for r in results),depiction_hint_sources=sum(bool(r['review_hints']['depiction']) for r in results),
        hints_are_review_queue_not_labels=True,training_labels_admitted=False,face_coverage_certified=False,
        source_roles_changed=False,new_test_inference=False,paid_model_api_calls=0)
    retain(a.output/(a.stage+'-summary.json'),summary);status.update(pid=None,stage='complete-with-invalid-content-hints' if summary['invalid_calls'] else 'complete-cached-content-hints',invalid_calls=summary['invalid_calls'],checked_unix=time.time());progress.write_text(json.dumps(status)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='calls'}))
    if summary['invalid_calls'] or summary['completed']!=summary['required']:raise ValueError('invalid/incomplete content calls retained for inspection')


if __name__=='__main__':main()
