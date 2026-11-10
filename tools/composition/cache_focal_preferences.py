#!/usr/bin/env python3
"""Cache frozen FocalNet descriptors for the existing TRAIN-fit/development bank."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from crop_fit import CropDataset
from artifact_identity import sha

MANIFEST = 'f911af240c7a130e710222c01e7b07ac2ee4978eb2962af19d708480512456b3'
CHECKPOINT = '82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a'


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['models', 'first_gate', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = Path(json.loads((args.first_gate/'inputs.json').read_text())['admitted_path'])
    checkpoint = args.models/'focalnet/focalnet-human.pt'
    if sha(manifest) != MANIFEST or sha(checkpoint) != CHECKPOINT:
        raise ValueError('registered label/initializer identity changed')
    old_path = args.first_gate/'feature-cache/features.pt'
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    if old['manifest_sha256'] != MANIFEST or len(old['sources']) != 80:
        raise ValueError('original preference bank differs')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.models/'focalnet-source/src'))
    from focalnet.ranking import load_human_checkpoint
    model, _ = load_human_checkpoint(checkpoint)
    model.requires_grad_(False).to(args.device).eval()
    captured = []
    hook = model.rank_head.register_forward_pre_hook(lambda _, values: captured.append(values[0].detach()))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    descriptors, scores, sources, uses, identities = [], [], [], [], []
    for use in ['fit', 'dev']:
        dataset = CropDataset(manifest, use, training=use == 'fit')
        if len(dataset) != (64 if use == 'fit' else 16):
            raise ValueError('registered source count changed')
        for i, source in enumerate(dataset.sources):
            index = len(sources)
            if old['sources'][index] != source or old['uses'][index] != use:
                raise ValueError('source/candidate/role identity differs from frozen bank')
            row = dataset[i]
            image = (row['image']-mean)/std
            x0, y0, x1, y1 = (row['valid_rect']*256).round().int().tolist()
            normalized = torch.zeros_like(image)
            normalized[:, y0:y1, x0:x1] = image[:, y0:y1, x0:x1]
            _, reference = model(normalized[None].to(args.device), row['boxes'][None].to(args.device),
                                 row['valid_rect'][None].to(args.device))
            if len(captured) != 1:
                raise ValueError('unexpected ranking descriptor hook count')
            vector = captured.pop()
            replay = model.rank_head(vector).squeeze(-1)
            captured.clear()
            if not torch.equal(replay, reference) or vector.shape != (1, 128, 62):
                raise ValueError('frozen descriptor replay differs')
            if not bool(torch.isfinite(vector).all()):
                raise ValueError('nonfinite descriptor')
            descriptors.append(vector[0].cpu())
            scores.append(reference[0].cpu())
            sources.append(source)
            uses.append(use)
            identities.append(dict(id=source['id'], tensor_sha256=source['tensor_sha256']))
            print(json.dumps(dict(completed=len(sources), id=source['id'], use=use)), flush=True)
    hook.remove()
    packed = old['packed']
    cache = dict(descriptors=torch.stack(descriptors), initial_ranks=torch.stack(scores),
                 conditions=torch.stack((packed['target_aspect'].log(), packed['mode']), dim=-1),
                 packed=packed, sources=sources, uses=uses,
                 rank_head_state=model.rank_head.cpu().state_dict(),
                 original_student_logits=old['initial_logits'],
                 original_cache_sha256=sha(old_path), manifest_sha256=MANIFEST,
                 checkpoint_sha256=CHECKPOINT, identities=identities)
    output = args.output/'features.pt'
    torch.save(cache, output)
    report = dict(sources=80, fit_sources=64, dev_sources=16, cache_sha256=sha(output),
                  shape=list(cache['descriptors'].shape), manifest_sha256=MANIFEST,
                  checkpoint_sha256=CHECKPOINT, frozen_descriptor_replay_exact=True,
                  diagnostic_maps_saved=False, canonical_holdouts_opened=False,
                  new_labels_admitted=False,
                  cuda_peak_allocated=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  cuda_peak_reserved=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
