#!/usr/bin/env python3
"""Compare the Python estimator with the captured official JS calculator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from sunburst_costs import output_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captured_js', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    source = a.captured_js.read_text()
    if 's={low:16,medium:24,high:48,xhigh:64,max:96}' not in source:
        raise ValueError('captured calculator settings changed; review oracle bindings')
    expression = source[source.index('_=') + 2:source.index(',v=')]
    cases = [{'model': 'gpt-image-2.6', 'quality': 'high', 'width': w, 'height': h}
             for w in range(16, 3841, 16) for h in range(16, 3841, 16)
             if max(w, h) <= 3 * min(w, h) and 655360 <= w * h <= 8294400]
    driver = '''const fs=require('node:fs'),vm=require('node:vm');
const d=JSON.parse(fs.readFileSync(0,'utf8'));
const f=vm.runInNewContext(d.expression,{l:{'gpt-image-2.6':{high:48}},c:{high:1}},{timeout:1000});
process.stdout.write(JSON.stringify(d.cases.map(x=>f(x))));'''
    values = json.loads(subprocess.check_output(['node', '-e', driver],
                       input=json.dumps({'expression': expression, 'cases': cases}).encode(), timeout=20))
    mismatches = [{'size': f'{c["width"]}x{c["height"]}', 'official_JS': value,
                   'python': output_tokens(f'{c["width"]}x{c["height"]}')}
                  for c, value in zip(cases, values)
                  if output_tokens(f'{c["width"]}x{c["height"]}') != value]
    report = {'valid_sizes_compared': len(cases), 'mismatches': mismatches,
              'maximum_official_high_output_tokens': max(values),
              'calculator_js_sha256': hashlib.sha256(source.encode()).hexdigest()}
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report))
    if mismatches or len(values) != len(cases):
        raise ValueError('calculator differs from captured official JS')


if __name__ == '__main__':
    main()
