#!/usr/bin/env python3
"""Summarize complete memory-study evidence, retaining every rejected packet."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifacts', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
root = args.artifacts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(prefix):
    counters = prefix.with_suffix('.stat').read_text()
    result = {'stat_sha256': digest(prefix.with_suffix('.stat'))}
    for key, pattern in {
        'cycles': r'^\s*([\d,]+)\s+(?:cpu_core/)?cycles(?:/|\s)',
        'instructions': r'^\s*([\d,]+)\s+(?:cpu_core/)?instructions(?:/|\s)',
        'minor_faults': r'^\s*([\d,]+)\s+minor-faults',
        'memory_bound_percent': r'([\d.]+) %\s+tma_memory_bound',
        'core_bound_percent': r'([\d.]+) %\s+tma_core_bound',
    }.items():
        match = re.search(pattern, counters, re.MULTILINE)
        if match:
            value = match[1].replace(',', '')
            result[key] = float(value) if key.endswith('percent') else int(value)
    usage = prefix.with_suffix('.time')
    if usage.exists():
        text = usage.read_text()
        result['time_sha256'] = digest(usage)
        result['peak_rss_kib'] = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text)[1])
        result['system_cpu_seconds'] = float(re.search(r'System time \(seconds\):\s*([\d.]+)', text)[1])
    return result


result = {
    'schema_version': 1,
    'library_source': '84d927d50efb4585d74f91541c04c92455d22152',
    'nn_source': '677d69be6822cccf57edcb0619d132b5addd0f86',
    'baseline_source': '4dd26ecc24ab68ecba133c7688d20ee6fdbec9fa',
    'baseline_nn_source': '084fadeb0b62309593dbc216f591ca6b73865a4f',
    'standalone_repeats': 101,
    'allocation_additional_images': 100,
    'standalone_i270': {},
    'diagnostic_counters_i270': {},
    'r7900x_1001_repeats': {},
    'confirmations': [],
}
for mode in ['std', 'alloc-only']:
    result['standalone_i270'][mode] = {}
    for arm in ['baseline', 'candidate']:
        directory = root / 'verified-final' / mode
        value = profile(directory / arm)
        value['executable_sha256'] = digest(directory / f'memory_{arm}')
        value['allocations'] = json.loads((directory / f'{arm}-allocations.json').read_text())
        assert value['allocations']['additional_images'] == 100
        result['standalone_i270'][mode][arm] = value
for path in sorted((root / 'final-counters').glob('*.stat')):
    result['diagnostic_counters_i270'][path.stem] = profile(path.with_suffix(''))
for mode in ['memory-profiles', 'memory-profiles-v4']:
    result['r7900x_1001_repeats'][mode] = {
        arm: profile(root / 'r7900x' / mode / arm) for arm in ['baseline', 'candidate']
    }
result['eight_worker_allocations'] = json.loads((root / 'profiles/final-eight-allocations.json').read_text())
rows = list(csv.DictReader((root / 'final-rectangles.tsv').open(), delimiter='\t'))
assert len(rows) == 1984
fields = ['max_map_abs_delta', 'roi_edge_map_px', 'crop_edge_map_px']
assert all(math.isfinite(float(row[field])) for row in rows for field in fields)
result['geometry'] = {
    'comparisons': len(rows),
    'maxima': {field: max(float(row[field]) for row in rows) for field in fields},
    'presence_mismatches': sum(row['presence_mismatch'] != 'false' for row in rows),
    'outside_budget': sum(row['within_budget'] != 'true' for row in rows),
    'sha256': digest(root / 'final-rectangles.tsv'),
}
for host, directory, logs in [
    ('i270', root / 'zenbench-i270', [root / name for name in [
        'paired-scaling.log', 'paired-scaling-confined.log', 'paired-scaling-root-scope.log',
        'paired-scaling-exact-scope.log', 'paired-scaling-other-widths.log']]),
    ('r7900x', root / 'r7900x/zenbench', [root / 'r7900x' / name for name in [
        'memory-paired-scaling.log', 'memory-paired-scaling-restaged.log', 'memory-paired-scaling-traced.log']]),
]:
    for log in logs:
        text = log.read_text()
        pointers = re.findall(r'aggregate evidence → (.*)', text)
        if not pointers:
            result['confirmations'].append({'host': host, 'log': str(log.relative_to(root)), 'valid': False, 'packets': 0, 'log_sha256': digest(log)})
        for name in pointers:
            pointer = json.loads((directory / 'runs' / Path(name).name).read_text())
            path = directory / 'objects' / f"{pointer['sha256']}.json.gz"
            assert digest(path) == pointer['sha256']
            value = json.loads(gzip.decompress(path.read_bytes()))
            children = value['replicates']
            clean = not value['unreliable'] and not value['incomplete'] and len(children) == 5
            means = {}
            failures = set()
            for child in children:
                clean &= not child['unreliable'] and not child['incomplete']
                clean &= child['cpu_placement']['status'] == 'enforced_uniform'
                for group in child['comparisons']:
                    clean &= group['completed_rounds'] == 10
                    for sample in group['samples']:
                        clean &= sample['gate_clean']
                        if not sample['gate_clean']:
                            failures.add(sample['gate_observation'].split(' state=')[0])
                    for index, bench in enumerate(group['benchmarks']):
                        label = bench if isinstance(bench, str) else bench['name']
                        means.setdefault(label, []).append(statistics.mean(s['compensated_ns'][index] / s['iterations'] for s in group['samples']))
            result['confirmations'].append({
                'host': host, 'log': str(log.relative_to(root)), 'valid': bool(clean),
                'packet_sha256': pointer['sha256'],
                'executable_sha256': value['execution']['binary_sha256'],
                'gate_waits': value['gate_waits'], 'failures': sorted(failures),
                'descriptive_mean_batch_ns': {k: statistics.mean(v) for k, v in means.items()},
            })
result['wrappers'] = {}
for name in ['verified-final-memory-study.log', 'final-worker-counters-geometry.log', 'final-eight-allocations.log',
             'border-inline-check.log', 'border-inline-parity-profile.log',
             'r7900x/memory-paired-scaling-restaged.log', 'r7900x/memory-paired-scaling-traced.log',
             'r7900x/memory-v4-profiling.log', 'r7900x/memory-v4-explicit-profiling.log']:
    text = (root / name).read_text()
    result['wrappers'][name] = next(line for line in reversed(text.splitlines()) if line.startswith('run-heavy: done'))
with args.output.open('x') as output:
    json.dump(result, output, indent=2)
    output.write('\n')
