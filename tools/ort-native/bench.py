#!/usr/bin/env python3
"""Run a bounded exploratory native stock/reduced comparison through zenbench."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--zenbench', default='zenbench')
a = p.parse_args()
r = a.root.resolve()
tool = Path(__file__).resolve().parent
ext = '.exe' if os.name == 'nt' else ''
build = r/'build'/'MinSizeRel'
def arm(name):
    return dict(id=name, program=str(build/('ort-native-'+name+ext)), args=['${output}'],
                revision='ORT1.31.1 '+name, timeout_seconds=60)
plan = dict(schema=1, baseline=arm('stock'), candidates=[arm('single')],
    cases=[dict(id='yunet640', inputs={'model': str(r/'model'/'yunet.ort'),
                                     'stock_library': str(build/'onnxruntime.dll')})],
    metric=dict(extract=dict(kind='prefix', prefix='elapsed_ms='), unit='ms', expected=3, divide_by=1),
    check=dict(kind='command', program=sys.executable, args=[str(tool/'check.py'),'${baseline}','${candidate}'], timeout_seconds=30),
    warmup='Three warm forwards; three ten-forward batches. Session construction, input and IO excluded. Unverified runner CPU placement.',
    blocks=4, aa=True, deadline_seconds=600)
path = r/'plan.json'
path.write_text(json.dumps(plan, indent=2)+'\n')
subprocess.run([a.zenbench, 'pair', str(path), '--output', str(r/'pair')], check=True,
               env=dict(os.environ, ZENBENCH_CPU_POLICY='unrestricted'))
