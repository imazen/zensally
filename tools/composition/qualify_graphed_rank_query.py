#!/usr/bin/env python3
"""Compare every selected head and development query before graph replay use."""
import argparse
import json
from pathlib import Path

import torch

from artifact_identity import sha
from fit_unsplash_ranks import evaluate, LocalCropRanker
from graphed_rank_query import GraphedRankQuery, evaluate_with_graph


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'plan', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if (sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'
            or sha(args.plan) != '6358383b2b7b2e1a3386334168fd9b457c9c2c17b2e37b96b74b003520696f45'):
        raise ValueError('frozen qualification identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
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
        raise ValueError('fixed development-only oracle required')
    data = {key:cache[key].cuda() for key in ['image', 'content', 'boxes', 'targets', 'pairs',
                                            'pair_valid', 'eligible', 'query_valid']}
    data['source_aspect'] = torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources], device='cuda')
    data['aspect'] = torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in cache['ratios'] for _ in range(2)], device='cuda')
    data['mode'] = torch.tensor([i % 2 for i in range(22)], dtype=torch.float32, device='cuda')
    donors = json.loads((args.cache.parent.parent/'fits/protocol.json').read_text())['donors']
    reports, context = [], None
    count = 0

    def verify(actual, expected):
        nonlocal count
        if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
            raise ValueError('query replay differs by FP32 bits')
        count += actual.numel()

    for binding in json.loads(args.plan.read_text())['models']:
        path = Path(binding['path'])
        if sha(path) != binding['sha256']:
            raise ValueError('selected weights changed')
        model = LocalCropRanker().cuda().eval()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
        model.configure(binding['arm'])
        model.eval()
        before = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        if context is None:
            context = GraphedRankQuery(model)
        actual_donors = torch.tensor(donors if binding['arm'] == 'permuted-image' else list(range(119)), device='cuda')
        expected = evaluate(model, data, indices, actual_donors, sources)
        actual = evaluate_with_graph(context, model, evaluate, data, indices, actual_donors, sources, verify=verify)
        if actual != expected or any(not torch.equal(v, before[k]) for k,v in model.state_dict().items()):
            raise ValueError('evaluation metrics/predictions or original state changed')
        if not torch.equal(rng[0], torch.get_rng_state()) or not torch.equal(rng[1], torch.cuda.get_rng_state()):
            raise ValueError('evaluation consumed RNG state')
        reports.append(dict(arm=binding['arm'], seed=binding['seed'], all_scores_bit_exact=True,
                            metrics_predictions_state_rng_unchanged=True))
        print(json.dumps(reports[-1]), flush=True)
    result = dict(models=reports, scalar_scores_compared=count, required_scores=12*25*22*128,
        complete=count == 12*25*22*128, test_sources_opened=0, training_math_changed=False,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (args.output/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
