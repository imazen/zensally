#!/usr/bin/env python3
"""Paired whole-development evaluation timing, with unchanged eager reductions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from artifact_identity import sha
from fit_unsplash_ranks import evaluate, LocalCropRanker
from graphed_rank_query import GraphedRankQuery, evaluate_with_graph


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'weights']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--backend', choices=['eager', 'graph'], required=True)
    args = parser.parse_args()
    if (sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'
            or sha(args.weights) != 'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df'):
        raise ValueError('frozen timing inputs differ')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sources = cache['sources']
    for source in sources:
        source['ratios'] = cache['ratios']
    indices = [i for i,s in enumerate(sources) if s['role'] == 'dev']
    if len(indices) != 25 or any(s['role'] == 'test' for s in sources):
        raise ValueError('fixed25-development timing views required')
    data = {key:cache[key].cuda() for key in ['image', 'content', 'boxes', 'targets', 'pairs',
                                            'pair_valid', 'eligible', 'query_valid']}
    data['source_aspect'] = torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources], device='cuda')
    data['aspect'] = torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in cache['ratios'] for _ in range(2)], device='cuda')
    data['mode'] = torch.tensor([i % 2 for i in range(22)], dtype=torch.float32, device='cuda')
    model = LocalCropRanker().cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
    donors = torch.arange(119, device='cuda')
    context = GraphedRankQuery(model) if args.backend == 'graph' else None

    def run():
        return (evaluate_with_graph(context, model, evaluate, data, indices, donors, sources)
                if context else evaluate(model, data, indices, donors, sources))

    expected = evaluate(model, data, indices, donors, sources)
    for _ in range(3):
        if run() != expected:
            raise ValueError('warm validation output changed')
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    first.record()
    for _ in range(30):
        actual = run()
    last.record()
    last.synchronize()
    if actual != expected:
        raise ValueError('timed validation output changed')
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    print('CHECK '+json.dumps(dict(digest=digest, sources=25, requests=550)))
    print('RESULT '+json.dumps(dict(evaluation_ms=first.elapsed_time(last)/30,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        scope='original25-source22-query development evaluation including encoder, finite checks, masks, CPU metrics and predictions; CUDA event interval includes host submission gaps; excludes model/cache loading and graph warmup')))


if __name__ == '__main__':
    main()
