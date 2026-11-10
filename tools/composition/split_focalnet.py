#!/usr/bin/env python3
"""Separate image-only graph work from candidate queries without changing weights."""
import argparse
import json
from collections import Counter
from pathlib import Path

import onnx

from fit_heads import sha


def frontier(graph):
    dependencies = {item.name: {item.name} for item in graph.input}
    dependencies.update({item.name: set() for item in graph.initializer})
    cut = set()
    for node in graph.node:
        if any(a.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS] for a in node.attribute):
            raise ValueError('this fixed-graph extractor does not support control-flow subgraphs')
        names = [name for name in node.input if name]
        if any(name not in dependencies for name in names):
            raise ValueError('graph is not topologically ordered or has unknown roots')
        roots = set().union(*(dependencies[name] for name in names))
        if roots & {'boxes', 'content'}:
            cut.update(name for name in names if dependencies[name] == {'image'})
        for output in node.output:
            dependencies[output] = roots
    if not cut or dependencies['importance'] != {'image'}:
        raise ValueError('expected image-independent candidate frontier and image-only importance')
    return sorted(cut)


def inventory(path):
    model = onnx.load(path)
    initializers = list(model.graph.initializer)
    return dict(file_bytes=path.stat().st_size, sha256=sha(path),
        nodes=len(model.graph.node), operators=dict(Counter(node.op_type for node in model.graph.node)),
        initializer_count=len(initializers),
        initializer_numeric_bytes=sum(onnx.numpy_helper.to_array(t).nbytes for t in initializers),
        inputs=[value.name for value in model.graph.input], outputs=[value.name for value in model.graph.output])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    cut = frontier(model.graph)
    analysis_outputs = list(dict.fromkeys(cut + ['importance']))
    onnx.utils.extract_model(str(args.model), str(args.output/'analysis.onnx'),
                             ['image'], analysis_outputs, check_model=True)
    onnx.utils.extract_model(str(args.model), str(args.output/'query.onnx'),
                             ['boxes', 'content'] + cut, ['crop_scores'], check_model=True)
    identity = dict(original=inventory(args.model), analysis=inventory(args.output/'analysis.onnx'),
        query=inventory(args.output/'query.onnx'), frontier=cut, analysis_outputs=analysis_outputs,
        onnx_version=onnx.__version__, source_sha256=sha(Path(__file__)),
        transformation='dependency frontier extraction only; no changed tensor weights or arithmetic operators',
        numerical_equivalence_measured=False, new_model_quality_claim=False)
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
