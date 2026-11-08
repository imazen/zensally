#!/usr/bin/env python3
"""Measure unified image-analysis plus crop-query costs at three thread policies."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['python', 'cache', 'weights', 'export', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name('bench_unsplash_student.py').resolve()
    inputs = dict(runner=str(runner), cache=str(args.cache.resolve()), weights=str(args.weights.resolve()))
    for name in ['encoder.onnx', 'query.onnx', 'qualification.json']:
        inputs[name] = str((args.export/name).resolve())
    for path in [runner.with_name('export_unsplash_ranker.py'), runner.with_name('artifact_identity.py'),
                 *(runner.parents[2]/'training'/name for name in ['local_rank_distillation.py', 'crop_regions.py',
                     'crop_ranker.py', 'model_v3ds.py', 'crop_fit.py'])]:
        inputs[path.name] = str(path)
    def arm(threads):
        return dict(id='analysis'+str(threads)+'-query1', program='/usr/bin/env',
            args=[str(args.python), '${runner}', '${cache}', '${weights}', str(args.export.resolve()),
                  '--analysis-threads', str(threads), '--phase', '${phase}'],
            env=dict(TMPDIR='/home/lilith/tmp', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            revision='qualified FP32 selected student; exact graphs and inputs in packet', timeout_seconds=120)
    plan = dict(schema=1, baseline=arm(1), candidates=[arm(2), arm(4)],
        cases=[dict(id=phase, fields=dict(phase=phase), inputs=inputs) for phase in ['one', 'bank']],
        metric=dict(extract=dict(kind='json', prefix='RESULT ', pointer='/pipeline_ms'), unit='ms', expected=1, divide_by=1),
        check=dict(kind='json', prefix='CHECK ', pointers=['/sources', '/requests', '/gate_violations']),
        warmup='one full25-photo sweep and same-student3-pixel numerical gate; five measured sweeps; one shared encoder per photo and1 or22 rank queries; persistent binding buffers',
        blocks=4, aa=True, deadline_seconds=900)
    with args.output.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
