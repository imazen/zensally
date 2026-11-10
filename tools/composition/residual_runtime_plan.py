#!/usr/bin/env python3
"""Register paired baseline/residual timings with a frozen numerical gate."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
    export=json.loads((root/'export.json').read_text());qualification=json.loads((root/'qualification.json').read_text())
    if qualification['gate_violations']:raise ValueError('native numerical gate required')
    binary=root/'binaries/residual-head'
    def arm(active):return dict(id='residual' if active else 'base',program=str(binary),args=['bench','${fixture}','residual' if active else 'baseline','${queries}','${repeats}',str(export['benchmark_query_offset'])],env=dict(TMPDIR=str(Path.home()/'tmp')),revision=qualification['binary_sha256'],timeout_seconds=120)
    inputs=dict(fixture=str(root/'hidden.bin'),model=str(root/'model.json'),qualification=str(root/'qualification.json'))
    plan=dict(schema=1,baseline=arm(False),candidates=[arm(True)],cases=[dict(id=name,fields=dict(queries=str(count),repeats=str(repeats)),inputs=inputs) for name,count,repeats in [('one',1,20000),('bank',22,1000)]],metric=dict(extract=dict(kind='json',prefix='RESULT ',pointer='/query_ms'),unit='ms',expected=1,divide_by=1),check=dict(kind='json',prefix='CHECK ',pointers=['/queries','/slots']),warmup='Fixture/model loading and one full query sweep excluded; persistent zenpredict buffers,128 candidate rows per query; first fixed DEV source, same binary includes both models.',blocks=4,aa=True,deadline_seconds=900)
    with (root/'timing-plan.json').open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
