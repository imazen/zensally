#!/usr/bin/env python3
"""Cache65 owned native ROI tensors and frozen visual head descriptors."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from artifact_identity import sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'runtime', 'models', 'resizer', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('owned resizer differs')
    suitability_path = args.root/'target-suitability.json'
    suitability = json.loads(suitability_path.read_text())
    manifest_path = args.root/'inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if suitability['input_manifest_sha256'] != sha(manifest_path) or not suitability['minimum_fit_family_support_met']:
        raise ValueError('registered teacher-support gate missing')
    selected = [r for r in suitability['rows'] if r['target_roi_normalized'] is not None]
    if len(selected) != 65 or sum(r['role'] == 'fit' for r in selected) != 53:
        raise ValueError('registered usable ROI cohort differs')
    views = {r['id']: r for r in manifest['views']}
    runtime = json.loads((args.runtime/'manifest.json').read_text())
    images = {r['id']: r for r in runtime['images']}
    args.output.mkdir(parents=True, exist_ok=False)
    tensors, content, geometry, sources = {}, {}, {}, []
    for source_id in dict.fromkeys(r['source_id'] for r in selected):
        rows = [r for r in selected if r['source_id'] == source_id]
        native = images[source_id]
        if not native['synthetic'] or sha(Path(native['rgb'])) != native['rgb_sha256']:
            raise ValueError('owned native source differs')
        specs, shapes = [], []
        for row in rows:
            view = views[row['id']]
            if view['source_rgb_sha256'] != native['rgb_sha256']:
                raise ValueError('ROI/source identity differs')
            l, t, r, b = view['native_bounds']
            width, height = r-l, b-t
            tw, th = max(1, round(width*256/max(width, height))), max(1, round(height*256/max(width, height)))
            specs.append(','.join(map(str, [l, t, width, height, tw, th])))
            shapes.append((tw, th, width, height))
        with Path(native['rgb']).open('rb') as stream:
            pixels = subprocess.check_output([str(args.resizer), str(native['width']), str(native['height']),
                str(native['width']*3), ';'.join(specs)], stdin=stream)
        if len(pixels) != sum(tw*th*3 for tw, th, _, _ in shapes):
            raise ValueError('owned neural-input resize length differs')
        offset = 0
        for row, (tw, th, width, height) in zip(rows, shapes, strict=True):
            image = np.zeros((256, 256, 3), np.uint8)
            ox, oy = (256-tw)//2, (256-th)//2
            size = tw*th*3
            image[oy:oy+th, ox:ox+tw] = np.frombuffer(pixels[offset:offset+size], np.uint8).reshape(th, tw, 3)
            path = args.output/(row['id']+'.rgb')
            path.write_bytes(image.tobytes())
            tensors[row['id']] = torch.from_numpy(image).permute(2, 0, 1).float()/255
            content[row['id']] = [ox/256, oy/256, (ox+tw)/256, (oy+th)/256]
            aspect = math.log(width/height)
            geometry[row['id']] = [aspect, math.tanh(aspect), tw/256, th/256]
            offset += size
    for row in selected:
        sources.append(dict(row, use=row['role'], image_tensor_sha256=sha(args.output/(row['id']+'.rgb')),
            geometry_training_eligible=row['role'] == 'fit', composition_training_eligible=False))
    image = torch.stack([tensors[r['id']] for r in selected])
    bounds = torch.tensor([content[r['id']] for r in selected])
    model_path = args.models/'focalnet/focalnet-human.pt'
    if sha(model_path) != '82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a':
        raise ValueError('pinned encoder differs')
    sys.path.insert(0, str(args.models/'focalnet-source/src'))
    from focalnet.ranking import load_human_checkpoint
    model, _ = load_human_checkpoint(model_path)
    model.cuda().eval()
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    vectors = []
    for start in range(0, len(image), 16):
        original = image[start:start+16].cuda()
        normalized = (original-mean)/std
        value = torch.zeros_like(normalized)
        for j, box in enumerate(bounds[start:start+16]):
            x0, y0, x1, y1 = (box*256).round().int().tolist()
            value[j, :, y0:y1, x0:x1] = normalized[j, :, y0:y1, x0:x1]
        fused = model.base.feature_pyramid(value)
        rank = model.rank_projection(fused)
        vector = torch.cat((F.adaptive_avg_pool2d(rank, (4, 4)).flatten(1),
            F.adaptive_avg_pool2d(fused, (2, 2)).flatten(1), fused.mean((-2, -1)),
            torch.zeros(len(value), 16, device='cuda')), -1)
        if vector.shape[1] != 512 or not bool(torch.isfinite(vector).all()):
            raise ValueError('invalid visual descriptors')
        vectors.append(vector.cpu())
    cache = dict(sources=sources, image=image, content=bounds, visual=torch.cat(vectors),
        geometry=torch.tensor([geometry[r['id']] for r in selected]),
        targets=torch.tensor([r['target_roi_normalized'] for r in selected]),
        suitability_sha256=sha(suitability_path), roi_manifest_sha256=sha(manifest_path),
        resizer_sha256=sha(args.resizer), encoder_sha256=sha(model_path),
        canonical_holdouts_opened=False, diagnostic_maps_saved=False,
        supervision='fit-only noisy ROI pose-head union; incomplete/uncertified', schema=1)
    torch.save(cache, args.output/'features.pt')
    identity = dict(cache_sha256=sha(args.output/'features.pt'), sources=sources,
        visual_shape=list(cache['visual'].shape), image_shape=list(image.shape),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        encoder_sha256=cache['encoder_sha256'], suitability_sha256=cache['suitability_sha256'])
    (args.output/'admission.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps({k:v for k,v in identity.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
