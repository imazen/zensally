#!/usr/bin/env python3
"""Frozen six-arm cache/thread scaling plan with exact rectangle checks."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
for name in ['models','split','cache','bank','output']:parser.add_argument(name,type=Path)
parser.add_argument('--python',required=True)
parser.add_argument('--mixed',action='store_true')
parser.add_argument('--spinning-study',action='store_true')
args=parser.parse_args()
if args.mixed and args.spinning_study:parser.error('choose one follow-up study')
runner=Path(__file__).with_name('cached_focalnet_bench.py').resolve()
helper=Path(__file__).with_name('cached_focalnet.py').resolve()
inputs=dict(model=str((args.models/'focalnet/focalnet-human.onnx').resolve()),
    analysis=str((args.split/'analysis.onnx').resolve()),query=str((args.split/'query.onnx').resolve()),
    cache=str(args.cache.resolve()),runner=str(runner),helper=str(helper),
    source_ids=str(runner.with_name('focalnet_bench.py')),
    source_crop=str((args.models/'focalnet-source/src/focalnet/crop.py').resolve()),
    source_imaging=str((args.models/'focalnet-source/src/focalnet/imaging.py').resolve()))
for file in sorted(args.bank.glob('*.json')):inputs['bank_'+file.stem]=str(file.resolve())
def arm(backend,threads,query_threads=None,no_spin=False):
    value=dict(id=f'{backend}-t{threads}'+('q'+str(query_threads) if query_threads else '')+('-no-spin' if no_spin else ''),program='/usr/bin/env',
        args=[args.python,'${runner}',str(args.models.resolve()),str(args.split.resolve()),'${cache}',str(args.bank.resolve()),'--backend',backend,'--threads',str(threads),'--repeats','3'],
        env=dict(TMPDIR='/home/lilith/tmp',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        revision='dependency-only graph split; exact worker/helper/models/banks hashed',timeout_seconds=120)
    if query_threads:value['args']+=['--query-threads',str(query_threads)]
    if no_spin:value['args']+=['--no-spin']
    return value
plan=dict(schema=1,baseline=arm('cached',2) if args.mixed else arm('full',1),
    candidates=[arm('cached',2,1),arm('cached',4,1),arm('cached',1,1)] if args.mixed else [arm('full',2),arm('full',4),arm('cached',1),arm('cached',2),arm('cached',4)],
    cases=[dict(id='six-scenes-eleven-ratios',inputs=inputs)],
    metric=dict(extract=dict(kind='json',prefix='RESULT ',pointer='/inference_ns'),unit='ns',expected=1,divide_by=1),
    check=dict(kind='json',prefix='CHECK ',pointers=['/digest','/requests']),
    warmup='one complete six-source sweep; verify 132 exact integer selected rectangles and actions outside timing; measure three sweeps; graph outputs reused in both paths',
    blocks=4,aa=True,deadline_seconds=1200)
if args.spinning_study:
    plan['baseline']=arm('cached',4,1)
    plan['candidates']=[arm('cached',4,4,True),arm('cached',4,1,True),arm('cached',4,4),arm('cached',2,1,True)]
with args.output.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
