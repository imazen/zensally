#!/usr/bin/env python3
"""One frozen-model comparison on new held-out local-rank references."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from analyze_unsplash_ranks import family_metrics, paired
from artifact_identity import sha
from fit_unsplash_ranks import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'plan', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.plan) != '6358383b2b7b2e1a3386334168fd9b457c9c2c17b2e37b96b74b003520696f45':
        raise ValueError('frozen pre-TEST model selection differs')
    if sha(args.cache) != '2f5a2d2fc1b61ce34d5181746bd12d46ad9973ae19f2ee0db4b1e7c2c5f16f6f':
        raise ValueError('qualified held-out reference cache differs')
    plan = json.loads(args.plan.read_text())
    for source in ['training/local_rank_distillation.py', 'training/crop_regions.py',
                   'training/crop_ranker.py', 'training/model_v3ds.py', 'tools/composition/fit_unsplash_ranks.py']:
        if Path(source).read_bytes() != subprocess.check_output(['git', 'show', plan['build_commit']+':'+source]):
            raise ValueError('student/evaluation source differs from selected implementation')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sources = cache['sources']
    if (cache['test_plan_sha256'] != sha(args.plan) or len(sources) != 25
            or any(s['role'] != 'test' for s in sources) or cache['training_eligible']):
        raise ValueError('frozen held-out-only reference contract differs')
    for source in sources:
        source['ratios'] = cache['ratios']
    data = {key:cache[key].cuda() for key in ['image', 'content', 'boxes', 'targets', 'pairs',
                                            'pair_valid', 'eligible', 'query_valid']}
    data['source_aspect'] = torch.tensor([s['native_size'][0]/s['native_size'][1] for s in sources], device='cuda')
    data['aspect'] = torch.tensor([int(r.split(':')[0])/int(r.split(':')[1]) for r in cache['ratios'] for _ in range(2)], device='cuda')
    data['mode'] = torch.tensor([i % 2 for i in range(22)], dtype=torch.float32, device='cuda')
    permutation = list(range(25))
    groups = defaultdict(list)
    for i, box in enumerate(cache['content']):
        groups[tuple((box*256).round().int().tolist())].append(i)
    generator = torch.Generator().manual_seed(31)
    for group in groups.values():
        order = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
        for i,j in zip(order, order[1:]+order[:1]):
            permutation[i] = j
    args.output.mkdir(parents=True, exist_ok=False)
    reports, all_rows = [], defaultdict(list)
    for binding in plan['models']:
        path = Path(binding['path'])
        if sha(path) != binding['sha256']:
            raise ValueError('frozen selected model changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['arm'] != binding['arm'] or state['seed'] != binding['seed']:
            raise ValueError('selected model metadata differs')
        model = LocalCropRanker().cuda()
        model.load_state_dict(state['model'])
        model.configure(binding['arm'])
        donors = torch.tensor(permutation if binding['arm'] == 'permuted-image' else list(range(25)), device='cuda')
        metrics, predictions = evaluate(model, data, list(range(25)), donors, sources)
        name = binding['arm']+'-seed'+str(binding['seed'])
        (args.output/(name+'.json')).write_text(json.dumps(predictions, indent=2)+'\n')
        reports.append(dict(arm=binding['arm'], seed=binding['seed'], weights_sha256=sha(path), metrics=metrics))
        all_rows[binding['arm']].extend(predictions)
        print(json.dumps(dict(completed=len(reports), required=12, arm=binding['arm'], seed=binding['seed'], metrics=metrics['all'])), flush=True)
    families = {arm:{scope:family_metrics(rows, scope) for scope in ['all', 'essential', 'secondary']}
                for arm,rows in all_rows.items()}
    if any(len(v['all']) != 22 for v in families.values()):
        raise ValueError('complete22 held-out-family support required')
    paired_results = {arm:{scope:paired(families['trainable-image'][scope], families[arm][scope])
        for scope in ['all', 'essential', 'secondary']} for arm in ['geometry-only', 'permuted-image', 'frozen-image']}
    summary = dict(sources=25, families=22, detector_person_positive_sources=8,
        cache_sha256=sha(args.cache), test_plan_sha256=sha(args.plan), models=reports,
        arms={arm:{key:statistics.mean(r['metrics']['all'][key] for r in reports if r['arm'] == arm)
                   for key in reports[0]['metrics']['all'] if key not in ['sources', 'families', 'valid_queries']}
              for arm in all_rows}, family_metrics=families, paired_trainable=paired_results,
        preselected_candidate=plan['candidate_for_export'], model_reselected=False, training_updated=False,
        test_permutation=permutation, test_images_moved_across_families=sum(s['family_id'] != sources[permutation[i]]['family_id']
            for i,s in enumerate(sources)), human_artistic_quality_measured=False, face_coverage_certified=False,
        canonical_holdout_pixels_opened=False, paid_model_api_calls=0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(arms=summary['arms'], paired_trainable_all={k:v['all'] for k,v in paired_results.items()},
        model_reselected=False, cuda_peak_allocated=summary['cuda_peak_allocated'], cuda_peak_reserved=summary['cuda_peak_reserved'])))


if __name__ == '__main__':
    main()
