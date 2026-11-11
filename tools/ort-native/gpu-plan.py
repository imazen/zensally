#!/usr/bin/env python3
"""Freeze four existing runtime fixtures and a bounded local GPU comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('fixtures', type=Path)
a = p.parse_args()
r = a.output.resolve()
r.mkdir(parents=True, exist_ok=True)
assets = r/'assets'
assets.mkdir()
data = (a.fixtures/'yunet.onnx').read_bytes()
assert hashlib.sha256(data).hexdigest() == '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
(assets/'yunet.onnx').write_bytes(data)
manifest = json.loads((a.fixtures/'inputs.json').read_text())
for f in manifest['fixtures']:
    data = (a.fixtures/f['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == f['sha256']
    (assets/f['file']).write_bytes(data)
x = np.empty(3*640*640, dtype='<f4')
state = 911
for i in range(x.size):
    state = (state*1664525+1013904223)&0xffffffff
    x[i] = state >> 24
x.tofile(assets/'diagnostic.f32')
(assets/'photo-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
tool = Path(__file__).resolve().parent
runtime = Path(ort.__file__).resolve().parent/'capi'
common = dict(model=str(assets/'yunet.onnx'), worker=str(tool/'gpu.py'), checker=str(tool/'check.py'),
              core=str(runtime/'libonnxruntime.so.1.31.1'),
              cuda_ep=str(runtime/'libonnxruntime_providers_cuda.so'), webgpu_ep=webgpu.get_library_path())
names = ['diagnostic.f32', *[f['file'] for f in manifest['fixtures']]]
def arm(name):
    return dict(id=name, program='/usr/bin/env',
                args=[sys.executable, str(tool/'gpu.py'), name, '${output}', '--model', '${model}', '--input', '${tensor}'],
                revision='ORT1.31.1 native '+name, timeout_seconds=120)
plan = dict(schema=1, baseline=arm('cpu'), candidates=[arm('cuda'), arm('webgpu')],
    cases=[dict(id=Path(n).stem, inputs=dict(common, tensor=str(assets/n))) for n in names],
    metric=dict(extract=dict(kind='prefix', prefix='elapsed_ms='), unit='ms', expected=3, divide_by=1),
    check=dict(kind='command', program='/usr/bin/env', args=[sys.executable, str(tool/'check.py'), '${baseline}', '${candidate}'], timeout_seconds=30),
    warmup='Three warm forwards; three ten-forward batches; Session.run returns completed CPU outputs. Includes upload/readback and output allocations; excludes construction/input/preprocessing/IO. Unverified CPU placement.',
    blocks=4, aa=True, deadline_seconds=1200,
    env={k: os.environ[k] for k in ['LD_LIBRARY_PATH','TMPDIR','XDG_RUNTIME_DIR'] if k in os.environ})
(r/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
print(r/'plan.json')
