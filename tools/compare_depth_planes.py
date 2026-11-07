#!/usr/bin/env python3
"""Compare row-packed little-endian u16 depth oracle captures."""
import argparse
from array import array
from collections import Counter
from pathlib import Path
import sys


def read(path):
    out = array('H')
    out.frombytes(Path(path).read_bytes())
    if sys.byteorder != 'little':
        out.byteswap()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('native')
    p.add_argument('oracle')
    p.add_argument('width', type=int)
    p.add_argument('height', type=int)
    args = p.parse_args()
    a, b = read(args.native), read(args.oracle)
    assert len(a) == len(b) == args.width * args.height
    differences = Counter(x-y for x, y in zip(a, b))
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print('samples', len(a), 'mismatches', len(bad), 'native_range', (min(a), max(a)), 'oracle_range', (min(b), max(b)))
    print('difference_histogram', differences.most_common(20))
    print('first_mismatches', [(i % args.width, i // args.width, a[i], b[i]) for i in bad[:20]])
    for name, index in [('flip_x', lambda x,y: y * args.width + args.width - 1-x), ('flip_y', lambda x,y: (args.height-1-y) * args.width+x), ('rotate_180', lambda x,y: (args.height-1-y) * args.width+args.width-1-x)]:
        n = sum(a[index(x, y)] != b[y * args.width + x] for y in range(args.height) for x in range(args.width))
        print(name, 'mismatches', n)


if __name__ == '__main__':
    main()
