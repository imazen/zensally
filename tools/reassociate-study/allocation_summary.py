#!/usr/bin/env python3
"""Classify heaptrack allocation-count differential stacks for the frozen w24 driver.

Stacks come from heaptrack_print --diff --flamegraph-cost-type allocations.
Counts include allocations and reallocations intercepted by heaptrack; they are
not page faults, bytes, live objects, or a claim about time spent allocating.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stacks", type=Path)
parser.add_argument("additional_images", type=int)
parser.add_argument("output", type=Path)
args = parser.parse_args()
assert args.additional_images > 0
counts = Counter()
for line in args.stacks.read_text().splitlines():
    stack, count = line.rsplit(" ", 1)
    if "forward_blis0" in stack:
        key = "dense_row_schedule"
    elif "BTensor3pad " in stack:
        key = "depthwise_padding"
    elif "BTensor7pad_par" in stack:
        key = "transpose_padding_pool_misses"
    elif "BTensor3cat" in stack:
        key = "concatenation_tensor"
    elif "spatial_mean" in stack:
        key = "spatial_means"
    elif "BTensor9to_planar" in stack:
        key = "caller_output"
    elif "BTensor7scratch" in stack:
        key = "activation_pool_misses"
    elif "VecRNt" in stack:
        key = "concatenation_references"
    elif "BTensor5zeros" in stack:
        key = "packed_input"
    elif "execute" in stack:
        key = "graph_bookkeeping"
    else:
        key = "other"
    counts[key] += int(count)
result = {
    "additional_images": args.additional_images,
    "total_allocation_calls": sum(counts.values()),
    "allocation_calls_per_image": sum(counts.values()) / args.additional_images,
    "categories": dict(sorted(counts.items())),
}
with args.output.open("x") as output:
    json.dump(result, output, indent=2)
    output.write("\n")
