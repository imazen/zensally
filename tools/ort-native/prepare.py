#!/usr/bin/env python3
"""Prepare one portable, fixed-optimization YuNet model for all native arms."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort, OptimizationStyle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
data = gzip.decompress((Path(__file__).resolve().parents[2]/'crates/zensally-tract/models/yunet_2023mar.onnx.gz').read_bytes())
if hashlib.sha256(data).hexdigest() != '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4':
    raise ValueError('unexpected source model')
model = a.output/'yunet.onnx'
model.write_bytes(data)
convert_onnx_models_to_ort(model, optimization_styles=[OptimizationStyle.Fixed], enable_type_reduction=True)
(a.output/'model-identity.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()}, indent=2)+'\n')
