#!/usr/bin/env python3
"""Bounded neural feature inputs and frozen selection masks for native query gates."""
import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch

from cached_focalnet import CachedContext, prepared
from fit_heads import sha
from focalnet_bench import IDS
from query_selection import eligible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['split','auxiliary','bank','output']:parser.add_argument(name,type=Path)
    parser.add_argument('--analysis-witnesses',nargs='+',default=IDS)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cache=args.auxiliary/'auxiliary.pt';aux=torch.load(cache,map_location='cpu',weights_only=False)
    witnesses=args.analysis_witnesses
    if not 1<=len(witnesses)<=6 or len(witnesses)!=len(set(witnesses)) or not set(witnesses)<=set(s['id'] for s in aux['sources']):
        raise ValueError('one to six explicit existing analysis witnesses required')
    context=CachedContext(args.split,threads=1)
    rows=[];hash_value=14695981039346656037;feature_payload_bytes=0
    with (args.output/'inputs.bin').open('xb') as inputs,(args.output/'reference.f32').open('xb') as reference:
        inputs.write(b'ZSCH001\0');inputs.write(struct.pack('<I',len(aux['sources'])))
        for i,source in enumerate(aux['sources']):
            image,content=prepared(aux,i);token=context.analyze(image,content,(source['id'],source['rgb_sha256']))
            if list(context.outputs)!=['transpose','importance']:raise ValueError('unexpected fixed frontier')
            features=context.outputs['transpose']
            feature_payload_bytes+=features.nbytes
            if features.shape!=(1,1024,17):raise ValueError('fixed 32x32x17 query features required')
            bank_path=args.bank/(source['id']+'.json');bank=json.loads(bank_path.read_text())
            features_path=args.output/(source['id']+'-features.npy');np.save(features_path,features,allow_pickle=False)
            image_path=None
            if source['id'] in witnesses:
                image_path=args.output/(source['id']+'-image.npy');np.save(image_path,image,allow_pickle=False)
            target_bank=args.output/(source['id']+'-bank.json');shutil.copy2(bank_path,target_bank)
            inputs.write(features.astype('<f4',copy=False).tobytes());inputs.write(content.astype('<f4',copy=False).tobytes());inputs.write(struct.pack('<I',len(bank['tasks'])))
            for task in bank['tasks']:
                boxes=np.zeros((1,128,4),np.float32)
                for j,candidate in enumerate(task['candidates']):boxes[0,j]=candidate['box']
                masks=[eligible(task,bank['width'],bank['height'],mode) for mode in ['minimal','maximal']]
                inputs.write(boxes.astype('<f4',copy=False).tobytes())
                for mask in masks:inputs.write(mask.tobytes())
                scores=context.score(boxes,token);reference.write(scores.astype('<f4',copy=False).tobytes())
                for mask in masks:
                    selected=np.flatnonzero(mask)
                    choice=int(selected[np.argmax(scores[0,selected])]) if len(selected) else 0xffffffff
                    for byte in struct.pack('<I',choice):hash_value=((hash_value^byte)*1099511628211)&0xffffffffffffffff
            rows.append(dict(id=source['id'],rgb_sha256=source['rgb_sha256'],content=content[0].tolist(),
                feature_file=features_path.name,feature_sha256=sha(features_path),
                image_file=image_path.name if image_path else None,image_sha256=sha(image_path) if image_path else None,
                bank_file=target_bank.name,bank_sha256=sha(target_bank),tasks=len(bank['tasks'])))
    manifest=dict(build_input_cache_sha256=sha(cache),split_identity_sha256=sha(args.split/'identity.json'),
        sources=rows,input_sha256=sha(args.output/'inputs.bin'),reference_sha256=sha(args.output/'reference.f32'),
        reference_choice_fnv64=str(hash_value),requests=sum(s['tasks']*2 for s in rows),
        neural_feature_cache=True,feature_payload_bytes=feature_payload_bytes,analysis_input_witnesses=witnesses,
        diagnostic_maps_saved=False,face_coverage_certified=False,new_labels_admitted=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['requests','reference_choice_fnv64','feature_payload_bytes','input_sha256']}))


if __name__=='__main__':main()
