#!/usr/bin/env python3
"""Warm complete cached-image crop pipeline timing with reusable ORT buffers."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import time
import torch

from artifact_identity import sha
from export_unsplash_ranker import LocalCropRanker


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def binding(graph, inputs, outputs):
    result = graph.io_binding()
    for name, value in inputs.items():
        result.bind_cpu_input(name, value)
    for name, value in outputs.items():
        result.bind_output(name, 'cpu', 0, np.float32, value.shape, value.ctypes.data)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'weights', 'export']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--analysis-threads', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--phase', choices=['one', 'bank'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':
        raise ValueError('frozen cache differs')
    qualification = json.loads((args.export/'qualification.json').read_text())
    if (sha(args.weights) != qualification['weights_sha256'] or
            any(r['gate_violations'] for r in qualification['runtime_gate'].values())):
        raise ValueError('qualified selected model required')
    for name in ['encoder', 'query']:
        if sha(args.export/(name+'.onnx')) != qualification['exports'][name]['sha256']:
            raise ValueError('qualified exported graph differs')
    encoder = session(args.export/'encoder.onnx', args.analysis_threads)
    query = session(args.export/'query.onnx', 1)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    model = LocalCropRanker().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
    frontier = dict(f8=np.empty((1, 16, 32, 32), np.float32), f16=np.empty((1, 16, 16, 16), np.float32))
    output = np.empty((1, 128), np.float32)
    views = []
    queries = list(range(22)) if args.phase == 'bank' else [0]
    for i, source in enumerate(cache['sources']):
        if source['role'] != 'dev':
            continue
        image = cache['image'][i:i+1]
        features = model.feature_maps(image)
        tasks = []
        for q in queries:
            ratio = cache['ratios'][q//2].split(':')
            values = dict(boxes=cache['boxes'][i:i+1, q//2].numpy(),
                source_aspect=np.array([source['native_size'][0]/source['native_size'][1]], np.float32),
                target_aspect=np.array([int(ratio[0])/int(ratio[1])], np.float32),
                mode=np.array([q % 2], np.float32), content=cache['content'][i:i+1].numpy())
            reference = model.scores(features, *(torch.from_numpy(v) for v in values.values())).numpy()[0]
            eligible = cache['eligible'][i, q].numpy()
            selected = int(np.where(eligible, reference, -np.inf).argmax())
            tasks.append(dict(binding=binding(query, dict(**frontier, **values), dict(scores=output)),
                              values=values, eligible=eligible, selected=selected))
        views.append(dict(binding=binding(encoder, dict(image=image.numpy()), frontier), image=image, tasks=tasks))
    if len(views) != 25:
        raise ValueError('all25 fixed development photos required')

    def run(verify=False):
        digest = 14695981039346656037
        for view in views:
            encoder.run_with_iobinding(view['binding'])
            for task in view['tasks']:
                query.run_with_iobinding(task['binding'])
                choice = int(np.where(task['eligible'], output[0], -np.inf).argmax())
                if verify:
                    boxes = task['values']['boxes'][0]
                    if not np.isfinite(output).all() or float(np.abs(boxes[choice]-boxes[task['selected']]).max()*128) > 3:
                        raise ValueError('timing runtime exceeds registered crop-edge gate')
                digest = ((digest ^ choice)*1099511628211) & 0xffffffffffffffff
        return digest

    expected = run(verify=True)
    start = time.perf_counter_ns()
    for _ in range(5):
        actual = run()
    elapsed = time.perf_counter_ns()-start
    if actual != expected:
        raise ValueError('repeated runtime selections changed')
    print('CHECK '+json.dumps(dict(sources=25, requests=25*len(queries), gate_violations=0)))
    print('RESULT '+json.dumps(dict(pipeline_ms=elapsed/5/25/1e6, choice_digest=str(actual),
        scope='one encoder per image plus'+str(len(queries))+' queries, retained ORT bindings and score buffers, fixed-mask selections; excludes decode, resize, input/model load and safety guards')))


if __name__ == '__main__':
    main()
