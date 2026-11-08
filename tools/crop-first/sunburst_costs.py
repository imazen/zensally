#!/usr/bin/env python3
"""Reproduce the captured official GPT Image 2.6 high-quality calculator."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

MODEL = 'gpt-image-2.6-sunburst'
MAX_HIGH_OUTPUT_TOKENS = 5930


def dimensions(size):
    w, h = map(int, size.split('x'))
    if (min(w, h) <= 0 or w % 16 or h % 16 or max(w, h) > 3840 or
            max(w, h) > 3 * min(w, h) or not 655360 <= w * h <= 8294400):
        raise ValueError('size outside captured calculator constraints')
    return w, h


def output_tokens(size):
    w, h = dimensions(size)
    # Preserve the JS division order and binary64 behavior around half-grid
    # boundaries. Exact rational rounding differs at four valid size pairs.
    short = round(48 / (max(w, h) / min(w, h)))
    numerator = 48 * short * (2_000_000 + w * h)
    return (numerator + 3_999_999) // 4_000_000


def row(size):
    w, h = dimensions(size)
    tokens = output_tokens(size)
    return {'size': size, 'ratio': str(Fraction(w, h)), 'pixels': w * h,
            'calculator_output_tokens': tokens,
            'calculator_output_usd': f'{tokens * 30 / 1_000_000:.5f}'}


def table():
    result = []
    for a, b in [(1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16), (2, 1), (1, 2), (3, 1), (1, 3)]:
        valid = []
        for scale in range(16, 3841, 16):
            size = f'{a * scale}x{b * scale}'
            try:
                valid.append(row(size))
            except ValueError:
                pass
        result.append({'requested_ratio': f'{a}:{b}',
                       'lowest_cost_exact_ratio': min(valid, key=lambda r: (r['calculator_output_tokens'], r['pixels']))})
    return {'model': MODEL, 'quality': 'high', 'input_tokens_excluded': True,
            'source': 'https://developers.openai.com/api/docs/guides/image-generation#calculating-costs',
            'exact_ratios': result,
            'practical_sizes': [row(s) for s in ['1024x1024', '2048x1024', '1024x2048',
                '1024x768', '768x1024', '1536x1152', '1152x1536', '1536x512',
                '512x1536', '3072x1024', '1024x3072', '3840x1280', '1280x3840']]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(table(), f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
