#!/usr/bin/env python3
"""Inspect the checked-in ONNX graph without importing a training runtime.

Field numbers follow the ONNX Model/Graph/Node/Tensor/Attribute protobuf schema.
This diagnostic never rewrites model bytes.
"""
import gzip
import struct
from collections import Counter
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


def fields(buf):
    offset = 0
    while offset < len(buf):
        key, offset = varint(buf, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = varint(buf, offset)
        elif wire == 2:
            size, offset = varint(buf, offset)
            value = buf[offset:offset + size]
            if len(value) != size:
                raise ValueError("truncated protobuf field")
            offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = buf[offset:offset + size]
            offset += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, value


def main():
    root = Path(__file__).resolve().parent.parent
    path = root / 'crates/zensally-tract/models/microsalnet_v3ds_w24.onnx.gz'
    model = gzip.decompress(path.read_bytes())
    graph = next(value for field, value in fields(model) if field == 7)
    nodes = []
    weights = []
    for field, value in fields(graph):
        if field == 1:
            node = {'inputs': [], 'outputs': [], 'attributes': {}}
            for tag, item in fields(value):
                if tag in (1, 2):
                    node['inputs' if tag == 1 else 'outputs'].append(item.decode())
                elif tag == 4:
                    node['op'] = item.decode()
                elif tag == 5:
                    attr = dict(fields(item))
                    name = attr[1].decode()
                    node['attributes'][name] = attr.get(3, attr.get(8, attr.get(2, attr.get(4, '?'))))
            nodes.append(node)
        elif field == 5:
            tensor = list(fields(value))
            name = next(item.decode() for tag, item in tensor if tag == 8)
            dims = [item for tag, item in tensor if tag == 1]
            raw = next((item for tag, item in tensor if tag == 9), b'')
            weights.append((name, dims, len(raw)))
    print('Operators:', dict(Counter(n['op'] for n in nodes)))
    print('Nodes:', len(nodes), 'Initializers:', len(weights))
    for node in nodes:
        print(node)
    print('Weights:')
    for weight in weights:
        print(weight)


if __name__ == '__main__':
    main()
