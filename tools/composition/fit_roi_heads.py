#!/usr/bin/env python3
"""Matched source-balanced frozen-feature ROI head controls; no crop promotion."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from composition_extent import ExtentHead, box_metrics
from roi_head_fit import per_roi_objective, source_weights
from artifact_identity import sha
from fit_extents import alias, retain


@torch.no_grad()
def evaluate(box, target, indices, sources, weights):
    if not bool(torch.isfinite(box).all()) or not bool(((box[:, 2:]-box[:, :2]) > 0).all()):
        raise ValueError('nonfinite or empty predicted head')
    truth = target[indices]
    _, iou = box_metrics(box, truth)
    losses = per_roi_objective(box, truth)
    mae, mse = (box-truth).abs().mean(-1), (box-truth).square().mean(-1)
    contains = ((box[:, :2] <= truth[:, :2]).all(-1) & (box[:, 2:] >= truth[:, 2:]).all(-1)).float()
    metrics = {k: float((v*weights).sum()) for k, v in dict(
        objective=losses, edge_mae=mae, edge_mse=mse, mean_iou=iou, contains_weak_target=contains).items()}
    rows = [dict(id=sources[i]['id'], source_id=sources[i]['source_id'], family_id=sources[i]['family_id'],
        role=sources[i]['use'], box=box[j].cpu().tolist(), weak_target=truth[j].cpu().tolist(),
        objective=float(losses[j]), edge_mae=float(mae[j]), edge_mse=float(mse[j]), iou=float(iou[j]),
        contains_weak_target=bool(contains[j]), face_coverage_certified=False) for j, i in enumerate(indices)]
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sources = cache['sources']
    fit = [i for i, s in enumerate(sources) if s['geometry_training_eligible']]
    dev = [i for i, s in enumerate(sources) if s['use'] == 'dev']
    if (len(sources), len(fit), len(dev)) != (65, 53, 12):
        raise ValueError('registered65ROI role contract differs')
    if (any(s['composition_training_eligible'] for s in sources)
            or any(sources[i]['use'] != 'fit' for i in fit)
            or {sources[i]['family_id'] for i in fit} & {sources[i]['family_id'] for i in dev}):
        raise ValueError('source-family/admission contract differs')
    fit_weight, dev_weight = source_weights(sources, fit), source_weights(sources, dev)
    visual = cache['visual']
    mean = (visual[fit]*fit_weight[:, None]).sum(0)
    std = ((visual[fit]-mean).square()*fit_weight[:, None]).sum(0).sqrt().clamp_min(.001)
    visual = ((visual-mean)/std).clamp(-10, 10)
    geometry = cache['geometry']
    permutation = list(range(len(sources)))
    rng = torch.Generator().manual_seed(31)
    groups = []
    for use in ['fit', 'dev']:
        for lower, upper in [(0, .5), (.5, .9), (.9, 1.1), (1.1, 2.), (2., float('inf'))]:
            indices = [i for i, s in enumerate(sources) if s['use'] == use
                       and lower <= math.exp(float(geometry[i, 0])) < upper]
            order = torch.randperm(len(indices), generator=rng).tolist()
            for i, j in enumerate(order):
                permutation[indices[i]] = indices[j]
            groups.append(dict(role=use, lower=lower, upper=upper if math.isfinite(upper) else None, indices=indices))
    inputs = dict(image=torch.cat((visual, geometry), -1).cuda(),
        geometry=torch.cat((torch.zeros_like(visual), geometry), -1).cuda(),
        permuted=torch.cat((visual[permutation], geometry), -1).cuda())
    target = cache['targets'].cuda()
    fw, dw = fit_weight.cuda(), dev_weight.cuda()
    protocol = dict(cache_sha256=sha(args.cache), seeds=[31, 137, 911], arms=list(inputs),
        epochs=200, batch=53, lr=.001, weight_decay=.01, clip=5., precision='FP32 TF32 off',
        loss='source-balanced5edgeL1+2GIoU', selection='minimum weak development objective includingepoch0',
        permutation=permutation, groups=groups, source_balanced=True, quality_model_promoted=False,
        moved_rois={role: sum(i != permutation[i] for i, s in enumerate(sources) if s['use'] == role)
                    for role in ['fit', 'dev']},
        moved_across_families={role: sum(s['family_id'] != sources[permutation[i]]['family_id']
            for i, s in enumerate(sources) if s['use'] == role) for role in ['fit', 'dev']})
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists():
        if json.loads((args.output/'protocol.json').read_text()) != protocol:
            raise ValueError('resume protocol differs')
    else:
        retain(args.output/'protocol.json', protocol)
    if not (args.output/'fit-mean-control.json').exists():
        control = (target[fit]*fw[:, None]).sum(0).expand(len(dev), 4)
        metrics, rows = evaluate(control, target, dev, sources, dw)
        retain(args.output/'fit-mean-control.json', dict(metrics=metrics, predictions=rows))
    reports = []
    for arm, features in inputs.items():
        for seed in protocol['seeds']:
            directory = args.output/(arm+'-seed'+str(seed))
            directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():
                reports.append(json.loads((directory/'summary.json').read_text()))
                continue
            torch.manual_seed(seed)
            model = ExtentHead().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, fused=True)
            history, best, best_epoch, start = [], float('inf'), None, 0
            if (directory/'latest.pt').exists():
                state = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
                if state['protocol'] != protocol:
                    raise ValueError('checkpoint protocol differs')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                history, best, best_epoch, start = state['history'], state['best'], state['best_epoch'], state['epoch']+1
            for epoch in range(start, 201):
                if epoch:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    value = (per_roi_objective(model(features[fit]), target[fit])*fw).sum()
                    if not bool(torch.isfinite(value)):
                        raise ValueError('nonfinite fit loss')
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    metrics, rows = evaluate(model(features[dev]), target, dev, sources, dw)
                history.append(dict(epoch=epoch, dev=metrics))
                improved = metrics['objective'] < best
                if improved:
                    best, best_epoch = metrics['objective'], epoch
                if improved or epoch % 25 == 0:
                    checkpoint = directory/f'epoch-{epoch:03}.pt'
                    with checkpoint.open('xb') as stream:
                        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch, protocol=protocol, history=history, best=best, best_epoch=best_epoch,
                            normalization_mean=mean, normalization_std=std), stream)
                    alias(directory/'latest.pt', checkpoint)
                    predictions = directory/f'predictions-{epoch:03}.json'
                    retain(predictions, rows)
                    if improved:
                        alias(directory/'best.pt', checkpoint)
                        alias(directory/'best-predictions.json', predictions)
                (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), arm=arm, seed=seed,
                    epoch=epoch, required_epochs=200, completed_runs=len(reports), required_runs=9,
                    checked_unix=time.time()))+'\n')
                if epoch % 50 == 0:
                    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 2*1024**3:
                        raise ValueError('fit-artifact2GiB budget exhausted; preserve completed states')
                    print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch, best_epoch=best_epoch, dev=metrics)), flush=True)
            report = dict(arm=arm, seed=seed, best_epoch=best_epoch, best_dev=history[best_epoch]['dev'],
                          parameters=sum(p.numel() for p in model.parameters()))
            retain(directory/'history.json', history)
            retain(directory/'summary.json', report)
            reports.append(report)
    retain(args.output/'summary.json', reports)
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete',
        completed_runs=9, required_runs=9, checked_unix=time.time()))+'\n')
    print(json.dumps(dict(completed_runs=9, cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved=torch.cuda.max_memory_reserved())))


if __name__ == '__main__':
    main()
