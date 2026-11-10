#!/usr/bin/env python3
"""Replace frozen ROI descriptors with the owned small encoder; preserve roles."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from artifact_identity import sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from crop_fit import TrainingModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'checkpoint', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.cache) != '1b56e11b7726576842f4efc322411c526b29b9bab5902f388150cacfc10d98e3':
        raise ValueError('registered parent ROI cache differs')
    if sha(args.checkpoint) != '910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':
        raise ValueError('registered owned encoder differs')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if cache['image'].shape != (65, 3, 256, 256) or cache['visual'].shape != (65, 512):
        raise ValueError('registered neural input shape differs')
    if any(s['composition_training_eligible'] for s in cache['sources']):
        raise ValueError('geometry targets cannot admit artistic labels')
    args.output.mkdir(parents=True, exist_ok=False)
    model = TrainingModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'])
    model.ranker.scorer = nn.Identity()
    model.cuda().eval().requires_grad_(False)
    vectors = []
    for start in range(0, 65, 16):
        maps = model.ranker.encode(cache['image'][start:start+16].cuda())
        visual = torch.cat([F.adaptive_avg_pool2d(value, (4, 4)).flatten(1) for value in maps], -1)
        if visual.shape[1] != 512 or not bool(torch.isfinite(visual).all()):
            raise ValueError('invalid frozen descriptors')
        vectors.append(visual.cpu())
    result = {**cache, 'visual': torch.cat(vectors), 'encoder_sha256': sha(args.checkpoint),
        'parent_cache_sha256': sha(args.cache), 'encoder_preprocessing': 'RGB0-1 zero letterbox padding'}
    for key in ['image', 'content', 'geometry', 'targets']:
        if not torch.equal(result[key], cache[key]):
            raise ValueError('frozen '+key+' changed')
    if result['sources'] != cache['sources']:
        raise ValueError('source identities/roles changed')
    torch.save(result, args.output/'features.pt')
    # This is a research state artifact, not a runtime graph or deployment size.
    torch.save({k: v.cpu() for k, v in model.ranker.state_dict().items()}, args.output/'encoder-state.pt')
    identity = dict(cache_sha256=sha(args.output/'features.pt'), encoder_sha256=result['encoder_sha256'],
        parent_cache_sha256=result['parent_cache_sha256'], preprocessing=result['encoder_preprocessing'],
        encoder_parameters=sum(p.numel() for p in model.parameters()),
        serialized_encoder_state_bytes=(args.output/'encoder-state.pt').stat().st_size,
        serialized_encoder_state_sha256=sha(args.output/'encoder-state.pt'),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        sources=65, fit_rois=53, dev_rois=12, unchanged_images_targets_geometry_sources=True,
        runtime_latency_measured=False, face_coverage_certified=False)
    (args.output/'admission.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(identity))


if __name__ == '__main__':
    main()
