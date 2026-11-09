#!/usr/bin/env python3
"""Cached native Florence region scout on owned, source-bound development inputs."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import Florence2ForConditionalGeneration, Florence2Processor

from artifact_identity import sha

PHRASES = {
    'MqOGVcLcJ3E':'A black leather wallet.',
    'v5C47rEShZ4':'A smartphone.',
    'kRnM-UJpK6I':'A book.',
    'nAMkUTfLjqI':'People in the passage.',
    '04dgurD3dn8':'A tree.',
    'FFPu5QDn40Y':'A boat and the moon.',
}
GROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'
TASKS = ['<MORE_DETAILED_CAPTION>', '<DENSE_REGION_CAPTION>', GROUNDING]
FEATURE_PHRASES = {
    'MqOGVcLcJ3E':'The embossed brand name on the wallet.',
    'v5C47rEShZ4':'The circular home button.',
    'kRnM-UJpK6I':'The cover of the book.',
    'nAMkUTfLjqI':'The central hanging plants.',
    '04dgurD3dn8':'The moss-covered branch at the tree fork.',
    'FFPu5QDn40Y':'The moon.',
}


def retain(path, value):
    def convert(item):
        if isinstance(item, set):
            return sorted(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('unsupported metadata type: '+type(item).__name__)
    payload = json.dumps(value, indent=2, default=convert)+'\n'
    if path.exists():
        if json.loads(path.read_text()) != json.loads(payload):
            raise ValueError('existing immutable metadata differs: '+str(path))
        return
    with path.open('x') as stream:
        stream.write(payload)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'registry', 'resizer', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--stage', choices=['probe','scout','features','primary56','supplementary15','photo87'], required=True)
    parser.add_argument('--source-plan', type=Path)
    args = parser.parse_args()
    primary = None
    if args.stage == 'primary56':
        if args.source_plan is None or sha(args.source_plan) != '6dc76f4d8730dcb91e17db659a5be330ebd18a88401b9caf2f594610590befa3':
            raise ValueError('registered primary56 source plan differs')
        primary = json.loads(args.source_plan.read_text())
        sources = primary['sources']
        fit = [s for s in sources if s['role'] == 'fit']
        dev = [s for s in sources if s['role'] == 'dev']
        if len(sources) != 56 or len(fit) != 48 or len(dev) != 8:
            raise ValueError('registered primary56 roles differ')
        if {s['family_id'] for s in fit} & {s['family_id'] for s in dev}:
            raise ValueError('primary56 families cross roles')
    elif args.stage == 'supplementary15':
        if args.source_plan is None or sha(args.source_plan) != 'a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70':
            raise ValueError('frozen supplementary development sources differ')
        primary = json.loads(args.source_plan.read_text())
        if len(primary['sources']) != 15 or any(s['role'] != 'supplementary_dev' or s['fit_eligible'] for s in primary['sources']):
            raise ValueError('fifteen development-only sources required')
    elif args.stage == 'photo87':
        if args.source_plan is None or sha(args.source_plan) != 'f597532945f20be5fb48770467affceabdee454b0051cd5698c6f13efe877d18':
            raise ValueError('frozen photographic subject queries differ')
        primary = json.loads(args.source_plan.read_text())
        sources = primary['sources']
        fit = [s for s in sources if s['role']=='fit']
        dev = [s for s in sources if s['role']=='dev']
        if len(sources)!=87 or len(fit)!=72 or len(dev)!=15 or any(s['training_eligible'] for s in sources):
            raise ValueError('registered72/15 photographic query sources required')
        if {s['family_id'] for s in fit} & {s['family_id'] for s in dev}:
            raise ValueError('photographic subject families cross roles')
    elif args.source_plan is not None:
        raise ValueError('source plan is only valid for a registered source-plan stage')
    if sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('fixed source-role registry differs')
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('pinned owned resizer differs')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['inputs','responses']:
        (args.output/name).mkdir(exist_ok=True)
    if (args.output/(args.stage+'-summary.json')).exists():
        raise ValueError('stage already complete; inspect cached results instead of regenerating')
    if args.stage == 'scout':
        probe = json.loads((args.output/'probe-summary.json').read_text())
        if probe['completed'] != 1 or probe['invalid_calls']:
            raise ValueError('a valid bounded wallet call must precede full scouting')
    if args.stage == 'features':
        scout = json.loads((args.output/'scout-summary.json').read_text())
        if scout['completed'] != 18 or scout['invalid_calls']:
            raise ValueError('completed valid object scout required before feature probes')
    assets = json.loads((args.assets/'asset-identities.json').read_text())
    if (assets['revision'] != '00921df66db728a9ceb750f5eca43e5c203a2051' or assets['remote_code_permitted']):
        raise ValueError('native-only pinned asset contract differs')
    for file in assets['files']:
        path = args.assets/file['path']
        if path.stat().st_size != file['bytes'] or sha(path) != file['sha256']:
            raise ValueError('local model asset changed')
    torch.set_num_threads(2)
    torch.manual_seed(31)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(pid=os.getpid(), stage='loading', required={'probe':1, 'scout':18, 'features':6, 'primary56':56, 'supplementary15':15,'photo87':87}[args.stage],
                  completed=0, checked_unix=time.time())
    progress = args.output/'progress.json'
    progress.write_text(json.dumps(status)+'\n')
    def finalize_status():
        if status['pid'] == os.getpid():
            status.update(pid=None, stage=status['stage']+'-failed-or-interrupted', checked_unix=time.time())
            progress.write_text(json.dumps(status)+'\n')
    atexit.register(finalize_status)
    model, loading = Florence2ForConditionalGeneration.from_pretrained(args.assets/'model',
        dtype=torch.float16, attn_implementation='eager', local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    retain(args.output/(args.stage+'-loading.json'), loading)
    if any(loading.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('checkpoint does not fully initialize the native model')
    model = model.eval().cuda()
    processor = Florence2Processor.from_pretrained(args.assets/'model', local_files_only=True,
                                                  trust_remote_code=False)
    versions = {name:importlib.metadata.version(name) for name in ['torch','transformers','huggingface-hub']}
    generation = dict(max_new_tokens=512, num_beams=3, do_sample=False, use_cache=True)
    identity = dict(repo=assets['repo'], revision=assets['revision'],
        weight_sha256=next(r['sha256'] for r in assets['files'] if r['path']=='model/model.safetensors'),
        asset_manifest_sha256=sha(args.assets/'asset-identities.json'), versions=versions,
        native_model_class=type(model).__name__, native_processor_class=type(processor).__name__,
        image_processor_class=type(processor.image_processor).__name__,
        model_parameters=sum(p.numel() for p in model.parameters()), dtype='float16', attention='eager',
        generation_config=model.generation_config.to_dict(), generation_overrides=generation,
        resizer_sha256=sha(args.resizer), preprocessing='owned zenresize Lanczos stretch768; processor resize/center-crop disabled',
        image_mean=processor.image_processor.image_mean, image_std=processor.image_processor.image_std,
        remote_code_executed=False, source_roles_changed=False, paid_model_api_calls=0)
    retain(args.output/(args.stage+'-model-identity.json'), identity)
    lookup = {s['source']['provider_id']:s for s in json.loads(args.registry.read_text())['sources']}
    if primary is not None:
        lookup = {s['source']['provider_id']:s for s in primary['sources']}
        if len(lookup) != {'supplementary15':15,'primary56':56,'photo87':87}[args.stage]:
            raise ValueError('duplicate registered provider IDs')
    keys = list(lookup) if primary is not None else (['MqOGVcLcJ3E'] if args.stage == 'probe' else list(PHRASES))
    tasks = TASKS if args.stage == 'scout' else [GROUNDING]
    rows = []
    for key in keys:
        source = lookup[key]
        allowed_roles = {'supplementary_dev'} if args.stage == 'supplementary15' else ({'fit','dev'} if primary is not None else {'dev'})
        if source['role'] not in allowed_roles or sha(Path(source['rgb'])) != source['rgb_sha256']:
            raise ValueError('registered source role/owned RGB differs')
        input_path, input_record = args.output/'inputs'/(key+'.rgb'), args.output/'inputs'/(key+'.json')
        input_contract = dict(source_id=source['id'], source_rgb_sha256=source['rgb_sha256'],
            working_size=source['working_size'], output_size=[768,768], resizer_sha256=sha(args.resizer))
        if input_record.exists():
            saved = json.loads(input_record.read_text())
            if saved['contract'] != input_contract or sha(input_path) != saved['rgb_sha256']:
                raise ValueError('cached owned input differs')
        else:
            w,h = source['working_size']
            with Path(source['rgb']).open('rb') as stream:
                rgb = subprocess.check_output([str(args.resizer), str(w), str(h), str(w*3),
                    f'0,0,{w},{h},768,768'], stdin=stream)
            if len(rgb) != 768*768*3:
                raise ValueError('owned model-input byte count differs')
            with input_path.open('xb') as stream:
                stream.write(rgb)
            retain(input_record, dict(contract=input_contract, rgb_sha256=sha(input_path)))
        image = np.frombuffer(input_path.read_bytes(), np.uint8).reshape(768,768,3).copy()
        for task in tasks:
            phrases = FEATURE_PHRASES if args.stage == 'features' else PHRASES
            phrase = source['query'] if primary is not None else phrases[key]
            prompt = task+(phrase if task == GROUNDING else '')
            contract = dict(identity=identity, source_id=source['id'], source_role=source['role'],
                native_size=source['native_size'], owned_input_sha256=sha(input_path), task=task, prompt=prompt)
            if primary is not None:
                contract['source_plan_sha256'] = sha(args.source_plan)
            # JSON persistence turns processor tuples into arrays. Normalize
            # before both hashing and equality checks so a genuine cache hit
            # does not depend on transient Python container types.
            contract = json.loads(json.dumps(contract))
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',',':')).encode()).hexdigest()
            response = args.output/'responses'/(digest+'.json')
            if response.exists():
                result = json.loads(response.read_text())
                if result['contract'] != contract or result['contract_sha256'] != digest:
                    raise ValueError('cached generation contract differs')
                reused = True
            else:
                status.update(stage='generating', source=source['id'], task=task,
                              contract_sha256=digest, checked_unix=time.time())
                progress.write_text(json.dumps(status)+'\n')
                inputs = processor(text=prompt, images=image, return_tensors='pt',
                                   do_resize=False, do_center_crop=False)
                expected = image.astype(np.float32).transpose(2,0,1)/255
                expected = (expected-np.array(identity['image_mean'],np.float32)[:,None,None])/np.array(identity['image_std'],np.float32)[:,None,None]
                actual = inputs['pixel_values'].float().numpy()
                if actual.shape != (1,3,768,768) or not np.isfinite(actual).all():
                    raise ValueError('unexpected normalized input tensor')
                normalization_error = float(np.abs(actual[0]-expected).max())
                if normalization_error > 1e-6:
                    raise ValueError('processor changed owned spatial inputs beyond pointwise normalization')
                device_inputs = {k:v.to(device='cuda', dtype=torch.float16 if v.is_floating_point() else v.dtype)
                                 for k,v in inputs.items()}
                generated = model.generate(**device_inputs, **generation)
                tokens = generated[0].cpu().tolist()
                raw = processor.batch_decode(generated, skip_special_tokens=False)[0]
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos,list) else [eos]
                result = dict(contract=contract, contract_sha256=digest, token_ids=tokens,
                    generated_text=raw, token_cap_reached=len(tokens)-1 >= generation['max_new_tokens'],
                    ended_with_eos=bool(tokens and tokens[-1] in eos), normalization_max_absolute_error=normalization_error,
                    training_eligible=False, face_coverage_certified=False)
                try:
                    result['parsed'] = processor.post_process_generation(text=raw, task=task,
                        image_size=tuple(source['native_size']))
                    result['postprocess_error'] = None
                except Exception as error:
                    result['parsed'] = None
                    result['postprocess_error'] = repr(error)
                # Preserve generation even if parsing is unsuitable. Cache failure
                # recovery must never issue another generation for identical input.
                retain(response, result)
                reused = False
            invalid = result['postprocess_error'] is not None or result['token_cap_reached'] or not result['ended_with_eos']
            rows.append(dict(id=source['id'], task=task, response=response.name, reused=reused, invalid=invalid,
                             parsed=result['parsed'], generated_text=result['generated_text']))
            status.update(completed=len(rows), checked_unix=time.time())
            progress.write_text(json.dumps(status)+'\n')
            print(json.dumps(rows[-1]), flush=True)
    summary = dict(stage=args.stage, completed=len(rows), required=status['required'],
        invalid_calls=sum(r['invalid'] for r in rows), cached_calls=sum(r['reused'] for r in rows), calls=rows,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        teacher_suitable_for_training=False, training_labels_admitted=False, face_coverage_certified=False,
        source_roles_changed=False, new_test_inference=False, paid_model_api_calls=0)
    retain(args.output/(args.stage+'-summary.json'), summary)
    status.update(pid=None, stage=args.stage+'-complete', checked_unix=time.time())
    progress.write_text(json.dumps(status)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'calls'}))
    if summary['invalid_calls'] or summary['completed'] != summary['required']:
        raise ValueError('bounded scout stage has invalid/incomplete calls; retained cached generation')


if __name__ == '__main__':
    main()
