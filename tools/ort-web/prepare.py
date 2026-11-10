#!/usr/bin/env python3
"""Prepare pinned YuNet and three owned-photo tensors for reduced ORT Web."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--flatbuffers-site', required=True)
    a = p.parse_args()
    if ort.__version__ != '1.31.1':
        raise ValueError('same ORT version as browser baseline required')
    sys.path.append(a.flatbuffers_site)
    from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort, OptimizationStyle
    a.output.mkdir(parents=True, exist_ok=False)
    compressed = Path('crates/zensally-tract/models/yunet_2023mar.onnx.gz')
    model = a.output/'yunet.onnx'
    model.write_bytes(gzip.decompress(compressed.read_bytes()))
    # No architecture-specific target: converter excludes x86 NCHWc layout.
    convert_onnx_models_to_ort(model, output_dir=a.output, optimization_styles=[OptimizationStyle.Fixed],
                             enable_type_reduction=True)
    parent = Path('/mnt/v/output/zensally/broader-people-risk/2026-11-09')
    sources = json.loads((parent/'plan.json').read_text())['sources']
    resizer = Path('/mnt/v/output/zensally/florence-region-suitability/2026-11-09/binaries/zensally-crop-preview-reference-20261108')
    fixtures = []
    for index in [0, 3, 13]:
        s = sources[index]
        raw = Path(s['rgb'])
        if sha(raw) != s['rgb_sha256']:
            raise ValueError('owned source pixels differ')
        w, h = s['working_size']
        resized = subprocess.check_output([str(resizer), str(w), str(h), str(w*3), f'0,0,{w},{h},640,640'], input=raw.read_bytes())
        if len(resized) != 640*640*3:
            raise ValueError('owned resize output shape differs')
        # Fixed diagnostic input, deliberately square-stretched. This tests
        # backend parity, not the existing Rust letterbox adapter's geometry.
        tensor = np.frombuffer(resized, dtype=np.uint8).reshape(640,640,3)[:,:,::-1].transpose(2,0,1).astype('<f4').copy()
        dest = a.output/f'input{index:02d}.f32'
        dest.write_bytes(tensor.tobytes())
        fixtures.append(dict(id=s['id'], file=dest.name, sha256=sha(dest), shape=[1,3,640,640],
                             source_rgb_sha256=s['rgb_sha256']))
    baseline = Path('/home/lilith/work/zen/zensr/tools/browser-demo/node_modules/onnxruntime-web/dist')
    vendor = a.output/'stock'; vendor.mkdir()
    for name in ['ort.wasm.bundle.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']:
        shutil.copyfile(baseline/name, vendor/name)
    manifest = dict(ort_version=ort.__version__, model_sha256=sha(model), compressed_model_sha256=sha(compressed),
                    fixtures=fixtures, inputs='three existing DEV sources; owned square resize; BGR FP32',
                    api_calls=0, test_inference=False, scope='raw backend output parity, not detector/crop qualification',
                    resizer_sha256=sha(resizer), stock={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in vendor.iterdir()})
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__': main()
