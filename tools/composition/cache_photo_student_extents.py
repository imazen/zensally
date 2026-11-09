#!/usr/bin/env python3
"""Extract frozen photographic-student descriptors for unchanged extent targets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
import torch
from torch.nn import functional as F

from local_rank_distillation import LocalCropRanker
from artifact_identity import sha


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache','auxiliary','weights','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35)
    if sha(args.weights) != 'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':
        raise ValueError('preselected photographic student differs')
    if sha(args.cache) != 'f2d6987d8d2290ae51ad43b5bf46d4ea99cf3c3c156e0c384b24ca2c5ba0944f':
        raise ValueError('registered target/descriptor cache differs')
    cache = torch.load(args.cache,map_location='cpu',weights_only=False)
    if sha(args.auxiliary) != cache['auxiliary_source_cache_sha256']:
        raise ValueError('original image tensor cache differs')
    auxiliary = torch.load(args.auxiliary,map_location='cpu',weights_only=False)
    if len(auxiliary['sources']) != 56:
        raise ValueError('exact registered source count required')
    for index,(left,right) in enumerate(zip(cache['sources'],auxiliary['sources'])):
        if any(left[k] != right[k] for k in ['id','family_id','use','rgb_sha256','image_tensor_sha256']):
            raise ValueError('source/pixel/family/role/tensor binding differs')
        tensor = auxiliary['image'][index]
        # The original admission hashes the HWC u8 canvas before conversion
        # to CHW unit-float. Check both the original byte identity and its
        # exact float reconstruction rather than hashing a different layout.
        canvas = (tensor.permute(1,2,0)*255).round().to(torch.uint8).contiguous()
        if (hashlib.sha256(canvas.numpy().tobytes()).hexdigest() != left['image_tensor_sha256']
                or not torch.equal(canvas.permute(2,0,1).float()/255,tensor)):
            raise ValueError('individual image tensor identity differs')
    state = torch.load(args.weights,map_location='cpu',weights_only=False)
    model = LocalCropRanker().eval().cuda()
    model.load_state_dict(state['model'])
    vectors = []
    for offset in range(0,56,16):
        maps = model.feature_maps(auxiliary['image'][offset:offset+16].cuda().contiguous(memory_format=torch.channels_last))
        vectors.append(torch.cat([F.adaptive_avg_pool2d(m,(4,4)).flatten(1) for m in maps],-1).cpu())
    visual = torch.cat(vectors)
    if visual.shape != (56,512) or not bool(torch.isfinite(visual).all()):
        raise ValueError('invalid frozen descriptor shape/values')
    result = dict(cache,visual=visual,encoder_sha256=sha(args.weights),
        parent_extent_cache_sha256=sha(args.cache),
        descriptor_origin='pre-TEST-selected photographic student; both feature maps avgpool4x4; encoder frozen')
    output = args.output/'extent-cache'
    output.mkdir(parents=True,exist_ok=False)
    torch.save(result,output/'features.pt')
    readback = torch.load(output/'features.pt',map_location='cpu',weights_only=False)
    if any(not torch.equal(readback[k],cache[k]) for k in ['geometry','targets','present']):
        raise ValueError('unchanged targets/geometry readback differs')
    metadata = {k:v for k,v in result.items() if k not in ['visual','geometry','targets','present']}
    metadata.update(cache_sha256=sha(output/'features.pt'),encoder_frozen=True,source_roles_changed=False,
        paid_model_api_calls=0,cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved())
    with (output/'admission.json').open('x') as stream:
        stream.write(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['cache_sha256','encoder_frozen','cuda_peak_allocated','cuda_peak_reserved']}))


if __name__ == '__main__':
    main()
