#!/usr/bin/env python3
"""Matched real-photo weak rank controls; new TEST composition scores stay closed."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from artifact_identity import sha
from fit_extents import retain

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import ARMS, LocalCropRanker, family_weights, objective


def alias(path, target):
    stage = path.with_name(path.name+'.new')
    stage.symlink_to(os.path.relpath(target.resolve(), path.parent.resolve()))
    stage.replace(path)


@torch.no_grad()
def encode(model, image, indices):
    chunks = [model.feature_maps(image[indices[start:start+16]]) for start in range(0, len(indices), 16)]
    return tuple(torch.cat([chunk[level] for chunk in chunks]) for level in range(2))


@torch.no_grad()
def evaluate(model, data, indices, donors, sources):
    model.eval()
    features = encode(model, data['image'], donors[indices])
    source_aspect = data['source_aspect'][indices]
    content = data['content'][indices]
    matrices, choices = [], []
    for query in range(22):
        q = torch.full((len(indices),), query, dtype=torch.long, device='cuda')
        boxes = data['boxes'][indices, q//2]
        raw = model.scores(features, boxes, source_aspect, data['aspect'][q], data['mode'][q], content)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError('nonfinite development scores')
        mask = data['eligible'][indices, q]
        selected = raw.masked_fill(~mask, -torch.inf).argmax(-1)
        target = data['targets'][indices, q]
        teacher = target.masked_fill(~mask, -torch.inf).argmax(-1)
        regret = target.gather(1, teacher[:, None]).squeeze(1)-target.gather(1, selected[:, None]).squeeze(1)
        pairs, pm = data['pairs'][indices, q], data['pair_valid'][indices, q]
        delta = raw.gather(1, pairs[..., 0])-raw.gather(1, pairs[..., 1])
        accuracy = (((delta > 0).float()+.5*(delta == 0).float())*pm).sum(-1)/pm.sum(-1).clamp_min(1)
        predicted_box = boxes[torch.arange(len(indices), device='cuda'), selected]
        reference_box = boxes[torch.arange(len(indices), device='cuda'), teacher]
        edge_motion = ((predicted_box-reference_box).abs()*128).amax(-1)
        loss = objective(raw, target, pairs, pm, mask, data['query_valid'][indices, q])
        matrices.append(torch.stack((regret, accuracy, (selected == teacher).float(),
            (edge_motion <= 3).float(), edge_motion, loss), -1))
        choices.append(torch.stack((selected, teacher), -1))
    values = torch.stack(matrices, 1).cpu()
    chosen = torch.stack(choices, 1).cpu()
    available = data['query_valid'][indices].cpu()
    fw = family_weights(sources, indices)
    metrics = {}
    names = ['teacher_regret', 'strong_pair_accuracy', 'exact_top1', 'within_3_map_pixels', 'max_edge_map_pixels', 'objective']
    for scope, queries in [('all', list(range(22))), ('essential', list(range(18))), ('secondary', list(range(18, 22)))]:
        valid = available[:, queries]
        if not bool(valid.any(-1).all()):
            raise ValueError('source has no registered ranking target in this ratio scope')
        per_source = (values[:, queries]*valid[..., None]).sum(1)/valid.sum(-1, keepdim=True)
        metrics[scope] = {name: float((per_source[:, k]*fw).sum()) for k,name in enumerate(names)}
        metrics[scope].update(sources=len(indices), families=len({sources[i]['family_id'] for i in indices}),
                              valid_queries=int(valid.sum()))
    predictions = [dict(id=sources[i]['id'], family_id=sources[i]['family_id'], query=q,
        ratio=sources[i]['ratios'][q//2], mode='minimal' if q % 2 == 0 else 'maximal',
        query_valid=bool(available[j, q]), selected=int(chosen[j, q, 0]), teacher=int(chosen[j, q, 1]),
        teacher_regret=float(values[j, q, 0]), strong_pair_accuracy=float(values[j, q, 1]),
        max_edge_map_pixels=float(values[j, q, 4]), face_coverage_certified=False)
        for j,i in enumerate(indices) for q in range(22)]
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'checkpoint', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    if sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':
        raise ValueError('registered local relative-rank cache differs')
    if sha(args.checkpoint) != '910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':
        raise ValueError('registered small encoder initializer differs')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sources = cache['sources']
    for source in sources:
        source['ratios'] = cache['ratios']
    fit = [i for i,s in enumerate(sources) if s['role'] == 'fit']
    dev = [i for i,s in enumerate(sources) if s['role'] == 'dev']
    if (len(sources), len(fit), len(dev)) != (119, 94, 25) or any(s['role'] == 'test' for s in sources):
        raise ValueError('fixed fit/development-only role contract differs')
    if {sources[i]['family_id'] for i in fit} & {sources[i]['family_id'] for i in dev}:
        raise ValueError('provider component crosses roles')
    data = {key:cache[key].cuda() for key in ['image', 'content', 'boxes', 'targets', 'pairs',
                                            'pair_valid', 'eligible', 'query_valid']}
    data['source_aspect'] = torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources], device='cuda')
    data['aspect'] = torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in cache['ratios'] for _ in range(2)], device='cuda')
    data['mode'] = torch.tensor([i % 2 for i in range(22)], dtype=torch.float32, device='cuda')
    phase = torch.tensor([int.from_bytes(hashlib.sha256(('query-phase-v1\0'+s['id']).encode()).digest()[:8], 'big') % 22 for s in sources])
    donors = list(range(119))
    groups = {}
    for i,s in enumerate(sources):
        shape = tuple((cache['content'][i]*256).round().int().tolist())
        groups.setdefault((s['role'], shape), []).append(i)
    generator = torch.Generator().manual_seed(31)
    for indices in groups.values():
        order = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        for i,j in zip(order, order[1:]+order[:1]):
            donors[i] = j
    protocol = dict(cache_sha256=sha(args.cache), initializer_sha256=sha(args.checkpoint),
        arms=ARMS, seeds=[31, 137, 911], epochs=220, source_batch=16, lr=.001, weight_decay=.01,
        clip=5., precision='FP32 TF32 off', objective='strong-pair logistic +0.25 masked score MSE',
        selection='minimum family-mean development regret, then higher pair accuracy, then earlier epoch',
        query_phases=phase.tolist(), donors=donors, source_families_balanced=True,
        new_test_sources_opened=0, face_coverage_certified=False, purpose=cache['purpose'],
        moved_across_families={role:sum(s['family_id'] != sources[donors[i]]['family_id']
            for i,s in enumerate(sources) if s['role'] == role) for role in ['fit', 'dev']})
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists():
        if json.loads((args.output/'protocol.json').read_text()) != protocol:
            raise ValueError('existing run protocol differs')
    else:
        retain(args.output/'protocol.json', protocol)
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model']
    fit_index = torch.tensor(fit, device='cuda')
    weight = torch.zeros(119, device='cuda')
    weight[fit_index] = family_weights(sources, fit).cuda()*len(fit)*22/data['query_valid'][fit_index].sum(-1)
    reports = []
    resume = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    if resume and (resume['protocol'] != protocol or resume['arm'] not in ARMS or resume['seed'] not in [31, 137, 911]):
        raise ValueError('resume identity/scientific protocol differs')
    for arm in ARMS:
        arm_donors = torch.tensor(donors if arm == 'permuted-image' else list(range(119)), device='cuda')
        for seed in [31, 137, 911]:
            directory = args.output/(arm+'-seed'+str(seed))
            directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():
                report = json.loads((directory/'summary.json').read_text())
                if sha(directory/'best.pt') != report['best_weights_sha256']:
                    raise ValueError('completed run weights are missing or changed')
                reports.append(report)
                continue
            attempt = 0
            while (directory/f'attempt-{attempt:03}').exists():
                attempt += 1
            run = directory/f'attempt-{attempt:03}'
            run.mkdir()
            torch.manual_seed(seed)
            model = LocalCropRanker(initial).cuda()
            model.configure(arm)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01, fused=True)
            history, best, best_path, start = [], None, None, 0
            visits = torch.zeros(119, 22, dtype=torch.int32)
            if resume and resume['arm'] == arm and resume['seed'] == seed:
                if resume['protocol'] != protocol:
                    raise ValueError('resume scientific protocol differs')
                model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
                history, best, best_path, start = resume['history'], tuple(resume['best']), resume['best_path'], resume['epoch']+1
                visits = resume['visits']
                torch.set_rng_state(resume['cpu_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
            frozen = None
            if arm == 'frozen-image':
                model.eval()
                frozen = encode(model, data['image'], list(range(119)))
            for epoch in range(start, 221):
                if epoch:
                    model.train_arm()
                    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                    for offset in range(0, len(fit), 16):
                        raw_indices = torch.tensor([fit[i] for i in order[offset:offset+16]])
                        query = (phase[raw_indices]+epoch-1) % 22
                        visits[raw_indices, query] += 1
                        keep = cache['query_valid'][raw_indices, query]
                        indices, q = raw_indices[keep].cuda(), query[keep].cuda()
                        if not len(indices):
                            continue  # No known target: no forward, optimizer or BN update.
                        optimizer.zero_grad(set_to_none=True)
                        features = (tuple(value[indices] for value in frozen) if frozen is not None
                                    else model.feature_maps(data['image'][arm_donors[indices]]))
                        raw = model.scores(features, data['boxes'][indices, q//2], data['source_aspect'][indices],
                            data['aspect'][q], data['mode'][q], data['content'][indices])
                        loss = objective(raw, data['targets'][indices, q], data['pairs'][indices, q],
                            data['pair_valid'][indices, q], data['eligible'][indices, q], data['query_valid'][indices, q])
                        value = (loss*weight[indices]).sum()/16
                        if not bool(torch.isfinite(value)):
                            raise ValueError('nonfinite fitting objective')
                        value.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                        optimizer.step()
                metrics, predictions = evaluate(model, data, dev, arm_donors, sources)
                history.append(dict(epoch=epoch, development=metrics))
                key = (metrics['all']['teacher_regret'], -metrics['all']['strong_pair_accuracy'], epoch)
                if best is None or key < best:
                    best = key
                    best_path = str((run/f'weights-{epoch:03}.pt').resolve())
                    torch.save(dict(model=model.state_dict(), epoch=epoch, arm=arm, seed=seed,
                        protocol=protocol, metrics=metrics), best_path)
                    retain(run/f'predictions-{epoch:03}.json', predictions)
                    alias(directory/'best.pt', Path(best_path))
                    alias(directory/'best-predictions.json', run/f'predictions-{epoch:03}.json')
                if epoch % 25 == 0 or epoch == 220:
                    checkpoint = run/f'checkpoint-{epoch:03}.pt'
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        arm=arm, seed=seed, protocol=protocol, history=history, best=best, best_path=best_path,
                        visits=visits, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), checkpoint)
                    alias(directory/'latest.pt', checkpoint)
                if epoch % 10 == 0:
                    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 4*1024**3:
                        raise ValueError('4GiB state-artifact budget exhausted; preserve completed files')
                    print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch, best_epoch=best[-1], development=metrics['all'])), flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), arm=arm, seed=seed,
                    epoch=epoch, required_epochs=220, completed_runs=len(reports), required_runs=12,
                    checked_unix=time.time()))+'\n')
            if not bool((visits[fit] == 10).all()) or bool(visits[dev].any()):
                raise ValueError('fixed source/query visit contract differs')
            report = dict(arm=arm, seed=seed, best_epoch=best[-1], best_dev=history[best[-1]]['development'],
                parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in parameters),
                best_weights_sha256=sha(Path(best_path)), attempts=attempt+1)
            retain(run/'history.json', history)
            retain(directory/'summary.json', report)
            reports.append(report)
    retain(args.output/'summary.json', reports)
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete', completed_runs=12,
        required_runs=12, checked_unix=time.time()))+'\n')
    print(json.dumps(dict(completed_runs=12, cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved())))


if __name__ == '__main__':
    main()
