#!/usr/bin/env python3
"""Isolated analysis/query comparisons with frozen whole-bank choice checks."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
for name in ['split','fixture','dispatched','scalar','output']:parser.add_argument(name,type=Path)
parser.add_argument('--python',required=True)
parser.add_argument('--analysis',action='store_true')
args=parser.parse_args()
runner=Path(__file__).with_name('query_phase_bench.py').resolve()
inputs=dict(runner=str(runner),helper=str(runner.with_name('cached_focalnet.py')),
            selection=str(runner.with_name('query_selection.py')),identity=str(runner.with_name('artifact_identity.py')),
            analysis=str((args.split/'analysis.onnx').resolve()),query=str((args.split/'query.onnx').resolve()),
            dispatched=str(args.dispatched.resolve()),scalar=str(args.scalar.resolve()))
for file in sorted(args.fixture.iterdir()):
    if file.is_file():inputs['fixture_'+file.name]=str(file.resolve())
def reference(threads):
    return dict(id=f'ort-{threads}',program='/usr/bin/env',
        args=[args.python,'${runner}',str(args.split.resolve()),str(args.fixture.resolve()),'--phase','analysis' if args.analysis else 'query','--threads',str(threads),'--repeats','5'],
        env=dict(TMPDIR='/home/lilith/tmp',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        revision='frozen graph/input/selection identities in packet',timeout_seconds=120)
def native(name,program,math,repeats):
    return dict(id=name,program=program,args=['bench',str((args.fixture/'inputs.bin').resolve()),str(repeats),math],
        env=dict(TMPDIR='/home/lilith/tmp'),revision='compiled model weights embedded; full-bank numerical gate retained separately',timeout_seconds=120)
plan=dict(schema=1,baseline=reference(1),
    candidates=[reference(2),reference(4)] if args.analysis else [
        native('native-dispatched',str(args.dispatched.resolve()),'ordered',5),
        native('native-algebraic',str(args.dispatched.resolve()),'algebraic',5),
        native('native-scalar',str(args.scalar.resolve()),'ordered',1)],
    cases=[dict(id='normalized-source-analysis' if args.analysis else 'cached-source-eleven-queries',inputs=inputs)],
    metric=dict(extract=dict(kind='json',prefix='RESULT ',pointer='/inference_ns'),unit='ns',expected=1,divide_by=1),
    check=dict(kind='json',prefix='CHECK ',pointers=['/digest','/requests']),
    warmup='one complete input sweep and exact numerical/selection checks outside timing; cached queries include fixed-mask selection and digest; native also builds summed-area context once/source. Query excludes encoder; analysis uses six fixed image witnesses, query all 56 neural-feature sources.',
    blocks=4,aa=True,deadline_seconds=1800)
with args.output.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
