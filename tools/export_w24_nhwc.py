#!/usr/bin/env python3
"""Generate crates/zensally-micro/src/w24_plan.rs; no deployed ONNX parser.

Reads the committed microsalnet_v3ds_w24.onnx.gz only. Refuses operators or
attributes outside the implemented subset. Run with --check to verify the
checked-in plan is up to date.
"""
import argparse
import gzip
import hashlib
import math
import re
import struct
from pathlib import Path


def varint(buf, offset):
    result = 0
    for shift in range(0, 70, 7):
        byte = buf[offset]
        offset += 1
        result |= (byte & 127) << shift
        if not byte & 128:
            return result, offset
    raise ValueError("invalid protobuf varint")


def fields(buf, base=0):
    pos = 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        tag, wire = key >> 3, key & 7
        if wire == 0:
            start = pos
            value, pos = varint(buf, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, pos = varint(buf, pos)
            else:
                size = 8 if wire == 1 else 4
            start = pos
            value = buf[pos:pos + size]
            if len(value) != size:
                raise ValueError('truncated protobuf')
            pos += size
        else:
            raise ValueError(f'unsupported wire {wire}')
        yield tag, value, base + start


def const_name(tensor_name):
    """enc3.1.conv.6.fc2.weight -> ENC3_1_CONV_6_FC2_WEIGHT"""
    return re.sub(r'[^0-9A-Za-z]+', '_', tensor_name).upper()


def generate(gz_bytes):
    model = gzip.decompress(gz_bytes)
    graph, base = next((v, o) for t, v, o in fields(model) if t == 7)
    raw_nodes, weights, tensors = [], [], {}
    for tag, value, offset in fields(graph, base):
        if tag == 1:
            raw_nodes.append(value)
        elif tag == 5:
            fs = list(fields(value, offset))
            name = next(v.decode() for t, v, _ in fs if t == 8)
            dims = [v for t, v, _ in fs if t == 1]
            dtype = next(v for t, v, _ in fs if t == 2)
            raw, off = next((v, o) for t, v, o in fs if t == 9)
            assert dtype in (1, 7), (name, dtype)
            assert len(raw) == (4 if dtype == 1 else 8) * math.prod(dims)
            tensors[name] = (len(weights), dims, dtype, raw)
            weights.append(
                f'    Weight {{ name: "{name}", dims: &{dims}, offset: {off}, bytes: {len(raw)} }},')
    slots = {'input': 0}
    nodes = []
    for node in raw_nodes:
        fs = list(fields(node))
        op = next(v.decode() for t, v, _ in fs if t == 4)
        ins = [v.decode() for t, v, _ in fs if t == 1]
        outs = [v.decode() for t, v, _ in fs if t == 2]
        assert len(outs) == 1
        attrs = {}
        for t, v, _ in fs:
            if t != 5:
                continue
            a = list(fields(v))
            name = next(v.decode() for t, v, _ in a if t == 1)
            vals = [v for t, v, _ in a if t == 8]
            attrs[name] = vals if vals else next(v for t, v, _ in a if t in (3, 4))
        inputs = [slots[ins[0]]]
        if op in ('Conv', 'ConvTranspose'):
            assert attrs.pop('auto_pad', b'NOTSET') == b'NOTSET'
            group = attrs.pop('group', 1)
            stride = attrs.pop('strides', [1, 1])
            pads = attrs.pop('pads', [0, 0, 0, 0])
            dil = attrs.pop('dilations', [1, 1])
            assert stride == [stride[0]] * 2 and pads == [pads[0]] * 4 and dil == [dil[0]] * 2
            w, b = tensors[ins[1]][0], tensors[ins[2]][0]
            assert tensors[ins[1]][2] == tensors[ins[2]][2] == 1
            if op == 'ConvTranspose':
                assert attrs.pop('output_padding', [0, 0]) == [0, 0] and group == 1 and dil == [1, 1]
                code = f'Op::Transpose {{ w: {w}, b: {b}, stride: {stride[0]}, pad: {pads[0]} }}'
            else:
                code = (f'Op::Conv {{ w: {w}, b: {b}, stride: {stride[0]}, pad: {pads[0]}, '
                        f'dilation: {dil[0]}, groups: {group} }}')
        elif op == 'ReduceMean':
            assert attrs.pop('keepdims', 1) == 1 and attrs.pop('noop_with_empty_axes', 0) == 0
            assert list(struct.unpack('<2q', tensors[ins[1]][3])) == [2, 3]
            code = 'Op::Mean'
        elif op == 'Concat':
            assert attrs.pop('axis') == 1
            inputs = [slots[n] for n in ins]
            code = 'Op::Concat'
        elif op in ('Add', 'Mul'):
            inputs = [slots[n] for n in ins]
            code = 'Op::' + op
        elif op in ('Relu', 'Sigmoid'):
            code = 'Op::' + op
        else:
            raise ValueError(f'unsupported operator {op}')
        assert not attrs, (op, attrs)
        nodes.append(f'    Node {{ inputs: &{inputs}, op: {code} }}, // {outs[0]}')
        slots[outs[0]] = len(nodes)
    assert len(nodes) == 82 and slots['output'] == 82 and len(weights) == 79

    idx = ['pub mod idx {']
    for i, name in enumerate(tensors):
        idx.append(f'    /// `{name}`')
        idx.append(f'    pub const {const_name(name)}: usize = {i};')
    idx.append('}')

    sha_gz = hashlib.sha256(gz_bytes).hexdigest()
    sha = hashlib.sha256(model).hexdigest()
    return ('// Generated by tools/export_w24_nhwc.py; do not edit.\n'
            f'// microsalnet_v3ds_w24.onnx.gz SHA-256: {sha_gz}\n'
            'use crate::w24::{Node, Op, Weight};\n\n'
            '/// Decompressed ONNX model byte length.\n'
            f'pub const MODEL_BYTES: usize = {len(model)};\n'
            '/// SHA-256 of the decompressed ONNX model.\n'
            f'pub const MODEL_SHA256: &str = "{sha}";\n\n'
            '/// Weight tensor index by ONNX initializer name.\n'
            + '\n'.join(idx) +
            '\n\n/// The 79 ONNX initializers in file order (f32 raw_data offsets\n'
            '/// into the decompressed model bytes).\n'
            'pub const WEIGHTS: &[Weight] = &[\n' + '\n'.join(weights) + '\n];\n\n'
            '/// The 82 ONNX nodes in topological order, for plan validation\n'
            '/// (execution is hand-scheduled in `w24.rs`).\n'
            'pub const NODES: &[Node] = &[\n' + '\n'.join(nodes) + '\n];\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    gz = (root / 'crates/zensally-tract/models/microsalnet_v3ds_w24.onnx.gz').read_bytes()
    output = generate(gz)
    path = root / 'crates/zensally-micro/src/w24_plan.rs'
    if args.check:
        assert path.read_text() == output, 'stale plan: run tools/export_w24_nhwc.py'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)


if __name__ == '__main__':
    main()
