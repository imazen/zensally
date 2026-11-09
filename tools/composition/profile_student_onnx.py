#!/usr/bin/env python3
"""Bound ORT node profiling to the predeclared six-photo review and22 requests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'export', 'review', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':
        raise ValueError('frozen profiling cache differs')
    args.output.mkdir(parents=True, exist_ok=False)
    selected = json.loads((args.review/'protocol.json').read_text())['selected_ids']
    if len(selected) != 6:
        raise ValueError('six predeclared sources required')
    sessions = {}
    qualification = json.loads((args.export/'qualification.json').read_text())
    for name in ['encoder', 'query']:
        path = args.export/(name+'.onnx')
        if sha(path) != qualification['exports'][name]['sha256']:
            raise ValueError('qualified runtime graph differs')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output/name)
        sessions[name] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    for source_id in selected:
        i = next(i for i,s in enumerate(cache['sources']) if s['id'] == source_id)
        source = cache['sources'][i]
        if source['role'] != 'dev':
            raise ValueError('development-only profiling required')
        f8, f16 = sessions['encoder'].run(None, dict(image=cache['image'][i:i+1].numpy()))
        for q in range(22):
            p, ratio_q = map(int, cache['ratios'][q//2].split(':'))
            result = sessions['query'].run(None, dict(f8=f8, f16=f16,
                boxes=cache['boxes'][i:i+1, q//2].numpy(), content=cache['content'][i:i+1].numpy(),
                source_aspect=np.array([source['native_size'][0]/source['native_size'][1]], np.float32),
                target_aspect=np.array([p/ratio_q], np.float32), mode=np.array([q % 2], np.float32)))[0]
            if not np.isfinite(result).all():
                raise ValueError('nonfinite profiled scores')
    reports = {}
    for name, session in sessions.items():
        path = Path(session.end_profiling())
        if path.stat().st_size > 64*1024**2:
            raise ValueError('64MiB trace budget exceeded; preserve trace')
        operations = defaultdict(lambda:dict(calls=0, duration_us=0))
        for row in json.loads(path.read_text()):
            if row.get('cat') == 'Node' and row['name'].endswith('_kernel_time'):
                value = operations[row['args']['op_name']]
                value['calls'] += 1
                value['duration_us'] += row['dur']
        reports[name] = dict(trace=path.name, trace_bytes=path.stat().st_size, trace_sha256=sha(path),
            operations=dict(sorted(operations.items(), key=lambda item:-item[1]['duration_us'])))
    summary = dict(source_ids=selected, requests=132, reports=reports,
        timing_instrumented=True, training_labels_produced=False, diagnostic_maps_saved=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
