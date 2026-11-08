#!/usr/bin/env python3
"""Registered matched feature distillation; teacher stays outside inference."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
import torch
from torch import nn

from composition_extent import ExtentHead, extent_objective
from composition_extent_hint import HintedExtentModel, hint_loss
from crop_fit import TrainingModel
from fit_extent_images import evaluate
from fit_extents import alias, retain
from fit_heads import sha


@torch.no_grad()
def errors(model, image, geometry, teacher, std, indices):
    _, prediction = model.forward_with_hint(image[indices], geometry[indices])
    return float(hint_loss(prediction, teacher[indices], std))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('auxiliary', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    out = args.root / 'extent-hint-fits'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cache_path = args.root / 'extent-cache/features.pt'
    teacher_path = args.root / 'focalnet-extent/extent-cache/features.pt'
    aux_path = args.auxiliary / 'auxiliary.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    teacher_cache = torch.load(teacher_path, map_location='cpu', weights_only=False)
    aux = torch.load(aux_path, map_location='cpu', weights_only=False)
    sources = cache['sources']
    if (sha(aux_path) != cache['auxiliary_source_cache_sha256'] or
            sha(args.checkpoint) != cache['encoder_sha256'] or
            teacher_cache['parent_extent_cache_sha256'] != sha(cache_path) or
            [s['id'] for s in aux['sources']] != [s['id'] for s in sources] or
            teacher_cache['sources'] != sources or
            not torch.equal(teacher_cache['targets'], cache['targets'])):
        raise ValueError('teacher/student/source identity differs')
    fit = [i for i, s in enumerate(sources) if s['primary_extent_training_eligible']]
    dev = [i for i, s in enumerate(sources) if s['use'] == 'dev' and s['primary_extent_present']]
    if {sources[i]['family_id'] for i in fit} & {sources[i]['family_id'] for i in dev}:
        raise ValueError('source families cross roles')
    permutation = json.loads((args.root / 'extent-fits/protocol.json').read_text())['source_permutation']
    mean = cache['visual'][fit].mean(0)
    std = cache['visual'][fit].std(0, unbiased=False).clamp_min(.001)
    teacher = teacher_cache['visual'][:, :256].cuda()
    hint_std = teacher[fit].std(0, unbiased=False).clamp_min(.1)
    geometry = cache['geometry'].cuda()
    target = cache['targets'].cuda()
    image = aux['image'].cuda().contiguous(memory_format=torch.channels_last)
    base = TrainingModel()
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'])
    base.ranker.scorer = nn.Identity()
    base.cuda().eval()
    protocol = {
        'cache_sha256': sha(cache_path), 'teacher_cache_sha256': sha(teacher_path),
        'auxiliary_sha256': sha(aux_path), 'checkpoint_sha256': sha(args.checkpoint),
        'arms': ['baseline', 'hint', 'permuted-hint'], 'seeds': [31, 137, 911],
        'epochs': 200, 'batch': 16, 'head_lr': .001, 'encoder_lr': .0001,
        'weight_decay': .01, 'clip': 5., 'hint_weight': .05,
        'hint': '16-channel rank pool4x4; training-only 16-to-16 pointwise adapter; fit-only std floor .1',
        'precision': 'FP32 TF32 off', 'batchnorm': 'frozen statistics; trainable affine',
        'loss': '5*edge L1 + 2*generalized IoU plus .05 standardized feature MSE for hint arms',
        'selection': 'minimum weak TRAIN-development extent objective, including epoch zero',
        'source_permutation': permutation, 'canonical_holdouts_opened': False,
        'face_coverage_certified': False,
        'source_sha256': {str(p.relative_to(Path.cwd())): sha(p) for p in [
            Path(__file__).resolve(), Path('training/composition_extent_hint.py').resolve(),
            Path('training/composition_extent_image.py').resolve(),
            Path('training/composition_extent.py').resolve()]},
    }
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('resume changes registered feature-hint protocol')
    else:
        retain(out / 'protocol.json', protocol)
    reports = []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            directory = out / (arm + '-seed' + str(seed))
            directory.mkdir(exist_ok=True)
            if (directory / 'summary.json').exists():
                reports.append(json.loads((directory / 'summary.json').read_text()))
                continue
            torch.manual_seed(seed)
            model = HintedExtentModel(copy.deepcopy(base.ranker), ExtentHead(), mean, std).cuda().eval()
            optimizer = torch.optim.AdamW([
                {'params': model.ranker.parameters(), 'lr': .0001},
                {'params': model.head.parameters(), 'lr': .001},
                {'params': model.hint_adapter.parameters(), 'lr': .001},
            ], weight_decay=.01, fused=True)
            history, best, best_epoch, start = [], float('inf'), None, 0
            teacher_indices = permutation if arm == 'permuted-hint' else list(range(len(sources)))
            if (directory / 'latest.pt').exists():
                state = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
                if state['protocol'] != protocol:
                    raise ValueError('checkpoint protocol differs')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                history, best, best_epoch = state['history'], state['best'], state['best_epoch']
                start = state['epoch'] + 1
            for epoch in range(start, 201):
                if epoch:
                    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(seed + epoch - 1)).tolist()
                    for offset in range(0, len(fit), 16):
                        indices = [fit[j] for j in order[offset:offset + 16]]
                        optimizer.zero_grad(set_to_none=True)
                        if arm == 'baseline':
                            box = model(image[indices], geometry[indices])
                            loss = extent_objective(box, target[indices])
                        else:
                            box, predicted_hint = model.forward_with_hint(image[indices], geometry[indices])
                            loss = extent_objective(box, target[indices]) + .05 * hint_loss(
                                predicted_hint, teacher[[teacher_indices[i] for i in indices]], hint_std)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('nonfinite feature-hint fitting loss')
                        loss.backward()
                        if arm == 'baseline' and any(p.grad is not None for p in model.hint_adapter.parameters()):
                            raise ValueError('inactive adapter received gradients')
                        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        optimizer.step()
                metrics, rows = evaluate(model, image, geometry, target, dev, dev, sources)
                item = {'epoch': epoch, 'dev': metrics}
                if epoch % 25 == 0:
                    item['own_teacher_hint_fit_mse'] = errors(model, image, geometry, teacher, hint_std, fit)
                    item['own_teacher_hint_dev_mse'] = errors(model, image, geometry, teacher, hint_std, dev)
                history.append(item)
                improved = metrics['objective'] < best
                if improved:
                    best, best_epoch = metrics['objective'], epoch
                if improved or epoch % 25 == 0:
                    checkpoint = directory / f'epoch-{epoch:03}.pt'
                    with checkpoint.open('xb') as stream:
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'epoch': epoch, 'arm': arm, 'seed': seed, 'protocol': protocol,
                                    'history': history, 'best': best, 'best_epoch': best_epoch}, stream)
                    alias(directory / 'latest.pt', checkpoint)
                    predictions = directory / f'predictions-epoch-{epoch:03}.json'
                    retain(predictions, rows)
                    if improved:
                        alias(directory / 'best.pt', checkpoint)
                        alias(directory / 'best-predictions.json', predictions)
                (out / 'progress.json').write_text(json.dumps({
                    'pid': os.getpid(), 'arm': arm, 'seed': seed, 'epoch': epoch,
                    'required_epochs': 200, 'completed_runs': len(reports), 'required_runs': 9,
                    'checked_unix': time.time()}))
                if epoch % 50 == 0:
                    print(json.dumps({'arm': arm, 'seed': seed, 'epoch': epoch,
                                      'dev': metrics, 'best_epoch': best_epoch}), flush=True)
            report = {'arm': arm, 'seed': seed, 'best_epoch': best_epoch,
                      'best_dev': history[best_epoch]['dev'],
                      'training_parameters': sum(p.numel() for p in model.parameters()),
                      'inference_parameters': sum(p.numel() for p in model.parameters()) -
                                              sum(p.numel() for p in model.hint_adapter.parameters())}
            retain(directory / 'history.json', history)
            retain(directory / 'summary.json', report)
            reports.append(report)
    retain(out / 'summary.json', reports)
    (out / 'progress.json').write_text(json.dumps({'pid': None, 'stage': 'complete',
        'completed_runs': len(reports), 'required_runs': 9, 'checked_unix': time.time()}))
    print(json.dumps({'completed_runs': len(reports), 'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                      'cuda_peak_reserved': torch.cuda.max_memory_reserved()}))


if __name__ == '__main__':
    main()
