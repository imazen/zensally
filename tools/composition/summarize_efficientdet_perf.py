#!/usr/bin/env python3
"""Extract measured counter/RSS evidence; refuse incomplete profile sets."""
import argparse
import json
from pathlib import Path
import re

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root/'manifest.json').read_text())
    rows = []
    for model in manifest['models']:
        path = args.root/'profiles-pidns'/model['id']
        stat_path, time_path = path.with_suffix('.stat'), path.with_suffix('.time')
        stat, timing = stat_path.read_text(), time_path.read_text()
        if re.search(r'Exit status: (\d+)', timing).group(1) != '0':
            raise ValueError('profile target failed')
        counters = {}
        for label in ['cycles', 'instructions', 'cache-misses', 'slots']:
            match = re.search(r'^\s*(\d+)\s+cpu_core/'+label+r'/', stat, re.M)
            if match is None:
                raise ValueError('counter unavailable: '+label)
            counters[label] = int(match.group(1))
        metrics = {name: float(value) for value, name in re.findall(
            r'#\s+([\d.]+) %\s+(tma_\w+)', stat)}
        if set(metrics) != {'tma_memory_bound', 'tma_core_bound'}:
            raise ValueError('both measured topdown metrics required')
        rows.append(dict(model=model['id'], counters=counters, topdown_percent=metrics,
            max_rss_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', timing).group(1)),
            stat_sha256=sha(stat_path), time_sha256=sha(time_path)))
    packet = json.loads((args.root/'timing-readable.json').read_text())
    observations = [o for invocation in packet['invocations'] for o in invocation['observations']]
    report = dict(rows=rows, profile_scope='800 calls/model, preloaded8RGB; whole target process counters/RSS include startup',
        topdown_denominator='pipeline slots, not elapsed time or DRAM latency',
        model_weight_bytes_separate=True, measured_dram_latency=False,
        paired_timing_exploratory=True, timing_cleanliness_certified=all(o['cleanliness_certified'] for o in observations),
        observed_heavy_process_counts=sorted({o['heavy_process_count'] for o in observations}))
    with (args.root/'profile-summary.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
