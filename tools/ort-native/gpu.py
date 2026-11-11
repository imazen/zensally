#!/usr/bin/env python3
"""Local native GPU probe. Timings finish with CPU-visible output tensors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('provider', choices=['cpu', 'cuda', 'webgpu'])
    p.add_argument('output', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--iterations', type=int, default=10)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    x = np.fromfile(a.input, dtype='<f4').reshape(1, 3, 640, 640)
    if not np.isfinite(x).all(): raise ValueError('nonfinite input')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CPUExecutionProvider']
    config = {}
    device = None
    if a.provider != 'cpu':
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    if a.provider == 'cuda':
        config = dict(device_id='0', use_tf32='0', gpu_mem_limit=str(1024**3),
                      cudnn_conv_algo_search='HEURISTIC', cudnn_conv_use_max_workspace='0')
        providers = [('CUDAExecutionProvider', config)]
    elif a.provider == 'webgpu':
        import onnxruntime_ep_webgpu as ep
        ort.register_execution_provider_library('zensally_webgpu', ep.get_library_path())
        devices = [d for d in ort.get_ep_devices()
                   if d.ep_name == ep.get_ep_name() and d.device.vendor_id == 0x10de]
        if len(devices) != 1: raise ValueError('exactly one NVIDIA WebGPU device required')
        device = dict(vendor_id=devices[0].device.vendor_id,
                      device_id=devices[0].device.device_id,
                      metadata=devices[0].device.metadata)
        config = dict(preferredLayout='NHWC', enableGraphCapture='0', validationMode='basic')
        options.add_provider_for_devices(devices, config)
        providers = None
    if a.profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(a.output/'ort-profile')
    session = ort.InferenceSession(a.model.read_bytes(), sess_options=options,
                                   providers=providers, enable_fallback=0)
    expected = dict(cpu='CPUExecutionProvider', cuda='CUDAExecutionProvider',
                    webgpu='WebGpuExecutionProvider')[a.provider]
    if expected not in session.get_providers(): raise ValueError('requested EP not active')
    feed = {session.get_inputs()[0].name: x}
    # Session.run returns CPU NumPy arrays, including completed device readback.
    def run(): return session.run(None, feed)
    for _ in range(3): run()
    for _ in range(3):
        begin = time.perf_counter_ns()
        for _ in range(a.iterations): run()
        print(f'elapsed_ms={(time.perf_counter_ns()-begin)/1e6/a.iterations:.12g}', flush=True)
    result = run()
    if len(result) != 12 or sum(v.size for v in result) != 134400:
        raise ValueError('incomplete YuNet output')
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in result):
        raise ValueError('invalid output dtype or values')
    (a.output/'outputs.f32').write_bytes(b''.join(v.astype('<f4', copy=False).tobytes() for v in result))
    evidence = dict(ort=ort.__version__, provider=a.provider, active=session.get_providers(),
                    options=config, device=device, model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                    input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                    outputs=[dict(name=m.name, shape=list(v.shape))
                             for m, v in zip(session.get_outputs(), result)],
                    graph_optimization='ORT_ENABLE_ALL', cpu_fallback_disabled=a.provider != 'cpu',
                    timing='synchronous host input to CPU-visible outputs, excludes construction/warmup/IO')
    if a.profile:
        profile = Path(session.end_profiling())
        events = json.loads(profile.read_text())
        placements = Counter((e['args'].get('provider'), e['args'].get('op_name'))
                             for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
        evidence['node_execution_counts'] = [{'provider': k[0], 'op': k[1], 'count': v}
                                              for k, v in sorted(placements.items())]
        if not any(k[0] == expected for k in placements): raise ValueError('no requested-provider kernel events')
    (a.output/'execution.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(provider=a.provider, outputs=len(result), scalars=134400)))


if __name__ == '__main__': main()
