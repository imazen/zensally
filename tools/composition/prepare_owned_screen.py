#!/usr/bin/env python3
"""Bind owned photo previews/tensors to unadmitted local runtime screening."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['acquired', 'plan', 'oracle', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sources = json.loads((args.acquired/'sources.json').read_text())
    plan = json.loads((args.plan/'sources.json').read_text())
    oracle = json.loads(args.oracle.read_text())
    if len(sources) != 101 or [s['id'] for s in sources] != [s['id'] for s in plan]:
        raise ValueError('registered acquired cohort differs')
    if [c['id'] for c in oracle['checks']] != [s['id'] for s in sources] or not all(
            all(c[key] for key in ['source_png_rgb_exact', 'preview_png_rgb_exact', 'tensor_hash_and_padding'])
            for c in oracle['checks']):
        raise ValueError('required independent acquisition readback missing')
    cache_root = args.output/'runtime-cache'
    cache_root.mkdir(parents=True, exist_ok=False)
    assets = args.output/'gallery-assets'
    assets.mkdir(exist_ok=False)
    rows, tensors, bounds, identities = [], [], [], []
    for source, original in zip(sources, plan):
        if any(source[key] != original[key] for key in ['id', 'family_id', 'sha256', 'use']):
            raise ValueError('acquired source/role differs from plan')
        if source['bucket'] != 'train' or source['use'] != 'screening' or source['training_eligible']:
            raise ValueError('only unadmitted TRAIN-screening sources allowed')
        directory = Path(source['directory'])
        raw = directory/'preview.rgb'
        width = max(1, round(source['width']*768/max(source['width'], source['height'])))
        height = max(1, round(source['height']*768/max(source['width'], source['height'])))
        if raw.stat().st_size != width*height*3:
            raise ValueError('owned preview shape differs')
        preview = directory/'preview.png'
        tensor_path = directory/'tensor.rgb'
        if sha(preview) != source['preview_sha256'] or sha(tensor_path) != source['tensor_sha256']:
            raise ValueError('owned preview/tensor identity changed')
        pixels = np.frombuffer(tensor_path.read_bytes(), np.uint8).copy().reshape(256, 256, 3)
        tensors.append(torch.from_numpy(pixels).permute(2, 0, 1).float()/255)
        bounds.append(source['valid_rect'])
        rows.append(dict(id=source['id'], family_id=source['family_id'], role='screening',
            rgb=str(raw), rgb_sha256=sha(raw), width=width, height=height,
            native_width=source['width'], native_height=source['height'],
            source_jxl_sha256=source['sha256'], source_rgb_sha256=source['rgb_sha256'],
            tensor_sha256=source['tensor_sha256'], synthetic=False, benchmark=False))
        identities.append(dict(source, rgb_sha256=source['rgb_sha256']))
        shutil.copy2(preview, assets/(source['id']+'.png'))
    cache = dict(image=torch.stack(tensors), packed=dict(valid_rect=torch.tensor(bounds)),
                 sources=identities, purpose='runtime screening only; no targets or admitted gradients',
                 plan_sha256=sha(args.plan/'sources.json'), oracle_sha256=sha(args.oracle),
                 diagnostic_maps_saved=False, training_eligible=False)
    torch.save(cache, cache_root/'auxiliary.pt')
    document = dict(schema=1, images=rows, acquisition_sha256=sha(args.acquired/'sources.json'),
                    plan_sha256=sha(args.plan/'sources.json'), oracle_sha256=sha(args.oracle),
                    source_roles_moved=False, canonical_holdouts_opened=False,
                    diagnostic_maps_saved=False, new_labels_admitted=False)
    with (args.output/'inputs.json').open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
    report = dict(sources=len(rows), cache_sha256=sha(cache_root/'auxiliary.pt'),
                  tensor_payload_bytes=cache['image'].numel()*cache['image'].element_size()+
                    cache['packed']['valid_rect'].numel()*cache['packed']['valid_rect'].element_size(),
                  purpose=cache['purpose'], training_eligible=False)
    (cache_root/'identity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
