#!/usr/bin/env python3
"""Registered frozen-encoder preference-head fits; canonical holdouts stay closed."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from composition_focal_head import FocalPreferenceHead
from crop_fit import loss
from fit_heads import predictions
from artifact_identity import sha

ARMS = ['visual', 'geometry', 'permuted']
SEEDS = [31, 137, 911]


def rank_counts(cache, indices, raw):
    counts = {name: dict(tasks=0, judged=0, accepted=0) for name in ['all', 'essential']}
    for row, index in enumerate(indices):
        source = cache['sources'][index]
        groups = {}
        for ci, candidate in enumerate(source['candidates']):
            groups.setdefault(candidate['task_id'], []).append(ci)
        for group in groups.values():
            ci = max(group, key=lambda c: float(raw[row, c, 0]))
            candidate = source['candidates'][ci]
            for name in ['all'] + (['essential'] if 1/3 <= candidate['aspect'] <= 3 else []):
                counts[name]['tasks'] += 1
                counts[name]['judged'] += int(candidate['weight'] > 0)
                counts[name]['accepted'] += int(candidate['weight'] > 0 and candidate['acceptable'])
    return counts


@torch.no_grad()
def evaluate(model, descriptors, conditions, packed, indices, cache):
    raw = model(descriptors[indices], conditions[indices])
    batch = {k: v[indices] for k, v in packed.items()}
    value, accept, rank = loss(raw, batch)
    scores = raw[..., 0]
    pairs, weights = batch['pairs'], batch['pairweights']
    delta = scores.gather(1, pairs[..., 0])-scores.gather(1, pairs[..., 1])
    valid = weights.sum(-1) > 0
    accuracies = (((delta > 0).float()+.5*(delta == 0).float())*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)
    cpu = raw.cpu()
    return dict(loss=float(value), accept_loss=float(accept), rank_loss=float(rank),
                source_mean_pair_accuracy=float(accuracies[valid].mean()),
                pair_sources=int(valid.sum()), rank_counts=rank_counts(cache, indices, cpu)), cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--cohort', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    fit = [i for i, use in enumerate(cache['uses']) if use == 'fit']
    dev = [i for i, use in enumerate(cache['uses']) if use == 'dev']
    if (len(fit), len(dev), len(cache['sources'])) != (64, 16, 80):
        raise ValueError('registered roles differ')
    cohort = None
    epochs = 60
    if args.cohort:
        cohort = json.loads(args.cohort.read_text())
        if (cohort['schema'], cohort['name'], cohort['cache_sha256']) != (
                1, 'existing-photo-preferences-v1', sha(args.cache)):
            raise ValueError('registered photographic cohort identity differs')
        admitted = {use: [] for use in ['fit', 'dev']}
        for member in cohort['members']:
            index = member['index']
            if not 0 <= index < 80 or any(cache['sources'][index][key] != member[key]
                    for key in ['id', 'family_id']) or cache['uses'][index] != member['use']:
                raise ValueError('cohort source/role differs')
            admitted[member['use']].append(index)
        fit, dev = admitted['fit'], admitted['dev']
        if (len(set(fit)), len(set(dev)), len(fit), len(dev)) != (15, 5, 15, 5):
            raise ValueError('registered photographic source counts differ')
        epochs = 240  # One batch/epoch matches the original 240 update budget.
    families = {}
    for source, use in zip(cache['sources'], cache['uses']):
        if source['family_id'] in families and families[source['family_id']] != use:
            raise ValueError('source family crosses gradient/selection roles')
        families[source['family_id']] = use
    descriptors = cache['descriptors'].to(args.device)
    conditions = cache['conditions'].to(args.device)
    packed = {k: v.to(args.device) for k, v in cache['packed'].items()}
    if descriptors.shape != (80, 128, 62) or conditions.shape != (80, 128, 2):
        raise ValueError('fixed descriptor/condition shapes required')
    weighted = (packed['weights'][fit]*packed['targets'][fit]).sum(-1)/packed['weights'][fit].sum(-1).clamp_min(1e-12)
    prior = min(1-1e-6, max(1e-6, float(weighted.mean())))
    prior_logit = math.log(prior/(1-prior))
    permutation = list(range(80))
    generator = torch.Generator().manual_seed(31)
    for indices in [fit, dev]:
        for i, j in enumerate(torch.randperm(len(indices), generator=generator).tolist()):
            permutation[indices[i]] = indices[j]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(arms=ARMS, seeds=SEEDS, epochs=epochs, batch=16, lr=1e-4,
                    weight_decay=.01, clip=5, dtype='FP32; TF32 disabled',
                    selection='minimum TRAIN-development loss including epoch0; all seeds retained',
                    cache_sha256=sha(args.cache), source_permutation=permutation,
                    acceptance_prior_fit_only=prior, canonical_holdouts_opened=False,
                    fit_indices=fit, dev_indices=dev,
                    cohort_sha256=sha(args.cohort) if args.cohort else None,
                    cohort_source_counts=[len(fit), len(dev)],
                    gradient_source_roles=['fit'], new_labels_admitted=False,
                    diagnostic_maps_saved=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    report = []
    for arm in ARMS:
        features = descriptors.clone()
        if arm == 'geometry':
            features[..., :51] = 0
        elif arm == 'permuted':
            features[..., :51] = descriptors[permutation, :, :51]
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = FocalPreferenceHead(cache['rank_head_state'], prior_logit).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01,
                                          fused=args.device == 'cuda')
            directory = args.output/(arm+'-seed'+str(seed))
            directory.mkdir()
            if arm == 'visual':
                with torch.no_grad():
                    for i in fit+dev:
                        raw = model(features[i:i+1], conditions[i:i+1])[0, :, 0].cpu()
                        if not torch.equal(raw, cache['initial_ranks'][i]):
                            raise ValueError('conditioned initializer changes frozen rank scores')
            history, best, best_epoch = [], float('inf'), None
            for epoch in range(epochs+1):
                if epoch:
                    model.train()
                    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(seed+epoch-1)).tolist()
                    for start in range(0, len(fit), 16):
                        indices = [fit[j] for j in order[start:start+16]]
                        batch = {k: v[indices] for k, v in packed.items()}
                        optimizer.zero_grad(set_to_none=True)
                        value, _, _ = loss(model(features[indices], conditions[indices]), batch)
                        if not bool(torch.isfinite(value)):
                            raise ValueError('nonfinite preference fitting loss')
                        value.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                        optimizer.step()
                model.eval()
                fit_metrics, _ = evaluate(model, features, conditions, packed, fit, cache)
                dev_metrics, raw = evaluate(model, features, conditions, packed, dev, cache)
                row = dict(epoch=epoch, fit=fit_metrics, dev=dev_metrics)
                history.append(row)
                improved = dev_metrics['loss'] < best
                if improved:
                    best, best_epoch = dev_metrics['loss'], epoch
                target = directory/f'epoch-{epoch:03}.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                epoch=epoch, arm=arm, seed=seed, protocol=protocol,
                                metrics=row), target)
                if improved:
                    stage = directory/'best.pt.new'
                    stage.symlink_to(target.name)
                    stage.replace(directory/'best.pt')
                    prediction = directory/f'predictions-epoch-{epoch:03}.json'
                    prediction.write_text(json.dumps(predictions(cache, dev, raw), indent=2)+'\n')
                    alias = directory/'best-predictions.json.new'
                    alias.symlink_to(prediction.name)
                    alias.replace(directory/'best-predictions.json')
                (directory/'history.json').write_text(json.dumps(history, indent=2)+'\n')
                (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), arm=arm,
                    seed=seed, epoch=epoch, required_epochs=epochs, completed_runs=len(report),
                    required_runs=9, last_checked_unix=time.time()))+'\n')
                if epoch % 20 == 0:
                    print(json.dumps(dict(arm=arm, seed=seed, **row, best_epoch=best_epoch)), flush=True)
            result = dict(arm=arm, seed=seed, best_epoch=best_epoch,
                          initial=history[0], selected=history[best_epoch],
                          parameters=sum(p.numel() for p in model.parameters()))
            report.append(result)
            (directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
            (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    complete = dict(stage='complete', pid=None, completed_runs=len(report), required_runs=9,
                    cuda_peak_allocated=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                    cuda_peak_reserved=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None,
                    last_checked_unix=time.time())
    (args.output/'progress.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
