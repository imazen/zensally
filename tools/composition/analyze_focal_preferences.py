#!/usr/bin/env python3
"""Recover selected preference heads and bootstrap paired development families."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from composition_focal_head import FocalPreferenceHead
from fit_heads import predictions
from artifact_identity import sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'fits', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    protocol = json.loads((args.fits/'protocol.json').read_text())
    if sha(args.cache) != protocol['cache_sha256']:
        raise ValueError('frozen preference cache changed')
    summary = json.loads((args.fits/'summary.json').read_text())
    if len(summary) != 9:
        raise ValueError('all registered preference fits are required')
    dev = protocol.get('dev_indices', [i for i, use in enumerate(cache['uses']) if use == 'dev'])
    if len(dev) != protocol.get('cohort_source_counts', [64, 16])[1]:
        raise ValueError('registered development role changed')
    if any(cache['uses'][index] != 'dev' for index in dev) or len(dev) != len(set(dev)):
        raise ValueError('checkpoint selection crosses or duplicates development roles')
    conditions = cache['conditions'].to(args.device)
    packed = {k: v[dev].to(args.device) for k, v in cache['packed'].items()}
    observations, checkpoints = {}, []
    for result in summary:
        arm, seed = result['arm'], result['seed']
        for variant in ['selected'] + (['initial'] if arm == 'visual' else []):
            epoch = result['best_epoch'] if variant == 'selected' else 0
            path = args.fits/f'{arm}-seed{seed}'/f'epoch-{epoch:03}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['protocol'] != protocol or checkpoint['arm'] != arm or checkpoint['seed'] != seed:
                raise ValueError('checkpoint identity differs')
            model = FocalPreferenceHead(cache['rank_head_state'], 0).to(args.device).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            descriptors = cache['descriptors'].clone().to(args.device)
            if arm == 'geometry':
                descriptors[..., :51] = 0
            elif arm == 'permuted':
                descriptors[..., :51] = descriptors[protocol['source_permutation'], :, :51]
            raw = model(descriptors[dev], conditions[dev])
            cpu = raw.cpu()
            recovered = predictions(cache, dev, cpu)
            expected = json.loads((path.parent/f'predictions-epoch-{epoch:03}.json').read_text())
            if [(r['task_id'], r['source_id'], r['candidate_id'], r['raw_rank_candidate_id']) for r in recovered] != [
                    (r['task_id'], r['source_id'], r['candidate_id'], r['raw_rank_candidate_id']) for r in expected]:
                raise ValueError('selected checkpoint does not recover stored choices/actions')
            weights = packed['weights']
            accept = (F.binary_cross_entropy_with_logits(raw[..., 1], packed['targets'], reduction='none')*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)
            pairweights = packed['pairweights']
            pairs = packed['pairs']
            delta = raw[..., 0].gather(1, pairs[..., 0])-raw[..., 0].gather(1, pairs[..., 1])
            rank = (F.softplus(-delta)*pairweights).sum(-1)/pairweights.sum(-1).clamp_min(1e-12)
            pairnum = (((delta > 0).float()+.5*(delta == 0).float())*pairweights).sum(-1)
            losses = (accept+rank).cpu().tolist()
            if abs(float((accept+rank).mean())-checkpoint['metrics']['dev']['loss']) > 1e-6:
                raise ValueError('checkpoint development loss does not reproduce')
            name = arm if variant == 'selected' else 'initial'
            for row, index in enumerate(dev):
                source = cache['sources'][index]
                counters = [0, 0, 0, 0]
                groups = {}
                for ci, candidate in enumerate(source['candidates']):
                    groups.setdefault(candidate['task_id'], []).append(ci)
                for group in groups.values():
                    ci = max(group, key=lambda j: float(cpu[row, j, 0]))
                    c = source['candidates'][ci]
                    judged = int(c['weight'] > 0)
                    accepted = judged*int(c['acceptable'])
                    counters[0] += accepted
                    counters[1] += judged
                    if 1/3 <= c['aspect'] <= 3:
                        counters[2] += accepted
                        counters[3] += judged
                pairvalid = float(pairweights[row].sum()) > 0
                accuracy = float(pairnum[row]/pairweights[row].sum()) if pairvalid else None
                observations.setdefault(name, {}).setdefault(source['family_id'], []).append(
                    [losses[row], accuracy, *counters])
            checkpoints.append(dict(arm=arm, seed=seed, variant=variant, epoch=epoch,
                                    path=str(path), sha256=sha(path), recovered_choices=True))
    families = sorted(observations['visual'])
    if any(sorted(data) != families for data in observations.values()):
        raise ValueError('bootstrap arms have different source families')
    means = {}
    pairfamilies = []
    for name, values in observations.items():
        rows = []
        for family in families:
            a = np.array([[np.nan if v is None else v for v in row] for row in values[family]])
            if np.isfinite(a[:, 1]).any() and not np.isfinite(a[:, 1]).all():
                raise ValueError('pair coverage differs between seeds')
            rows.append([a[:, 0].mean(), a[:, 1].mean(), *a[:, 2:].mean(0)])
        means[name] = np.array(rows)
        current = np.flatnonzero(np.isfinite(means[name][:, 1])).tolist()
        if pairfamilies and current != pairfamilies:
            raise ValueError('pair coverage differs between arms')
        pairfamilies = current
    rng = np.random.default_rng(31)
    samples = rng.integers(0, len(families), (5000, len(families)))
    pairsamples = rng.choice(pairfamilies, (5000, len(pairfamilies)), replace=True)
    def metrics(a, sample=None, pairsample=None):
        rows = a if sample is None else a[sample]
        pairrows = a[pairfamilies, 1] if pairsample is None else a[pairsample, 1]
        return dict(loss=rows[..., 0].mean(-1), pair_accuracy=pairrows.mean(-1),
                    rank_accepted_fraction=rows[..., 2].sum(-1)/rows[..., 3].sum(-1),
                    essential_rank_accepted_fraction=rows[..., 4].sum(-1)/rows[..., 5].sum(-1))
    observed = {name: {k: float(v) for k, v in metrics(a).items()} for name, a in means.items()}
    boot = {name: metrics(a, samples, pairsamples) for name, a in means.items()}
    comparisons = {}
    for control in ['initial', 'geometry', 'permuted']:
        comparisons[control] = {key: dict(delta=observed['visual'][key]-observed[control][key],
            percentile95=np.quantile(boot['visual'][key]-boot[control][key], [.025, .975]).tolist())
            for key in observed['visual']}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(families=len(families), pair_families=len(pairfamilies), bootstrap_samples=5000,
                  seed_averaging='within source family before bootstrap', observed=observed,
                  visual_minus_control=comparisons, checkpoints=checkpoints,
                  cache_sha256=sha(args.cache), confirmatory=False,
                  limitation='same development sources select checkpoints; weak model labels; no independent artistic-quality inference',
                  canonical_holdouts_opened=False, new_labels_admitted=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'source-family-means.json').write_text(json.dumps(dict(
        families=families, columns=['loss', 'pair_accuracy', 'accepted', 'judged',
                                   'essential_accepted', 'essential_judged'],
        arms={name: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]
              for name, a in means.items()}), indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['families', 'pair_families', 'observed', 'visual_minus_control']}, indent=2))


if __name__ == '__main__':
    main()
