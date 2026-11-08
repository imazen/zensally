#!/usr/bin/env python3
"""Declare matched eager/graph whole-validation timing with an A/A arm."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['python', 'cache', 'weights', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name('bench_graphed_rank_query.py').resolve()
    inputs = dict(runner=str(runner), cache=str(args.cache.resolve()), weights=str(args.weights.resolve()))
    for path in [runner.with_name('graphed_rank_query.py'), runner.with_name('fit_unsplash_ranks.py'),
                 *(runner.parents[2]/'training'/name for name in ['local_rank_distillation.py',
                     'crop_regions.py', 'crop_ranker.py', 'model_v3ds.py', 'crop_fit.py'])]:
        inputs[path.name] = str(path)
    def arm(name):
        return dict(id=name, program='/usr/bin/env', args=[str(args.python), '${runner}', '${cache}', '${weights}', '--backend', name],
            env=dict(TMPDIR='/home/lilith/tmp', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
            revision='fixed pre-TEST student; exact source/cache/weight identities in packet', timeout_seconds=120)
    plan = dict(schema=1, baseline=arm('eager'), candidates=[arm('graph')],
        cases=[dict(id='all25-development-sources22-queries', inputs=inputs)],
        metric=dict(extract=dict(kind='json', prefix='RESULT ', pointer='/evaluation_ms'), unit='ms', expected=1, divide_by=1),
        check=dict(kind='json', prefix='CHECK ', pointers=['/digest', '/sources', '/requests']),
        warmup='three full evaluations; graph capture/replay warmed before timing;30 repeated whole evaluations using CUDA events with explicit end synchronization',
        blocks=4, aa=True, deadline_seconds=600)
    with args.output.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
