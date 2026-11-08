#!/usr/bin/env python3
"""Require exact matched geometry controls before comparing visual encoders."""
import argparse
import json
from pathlib import Path

from analyze_roi_heads import paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('focal', type=Path)
    parser.add_argument('small', type=Path)
    args = parser.parse_args()
    checks = {}
    for seed in [31, 137, 911]:
        directory = 'geometry-seed'+str(seed)
        for name in ['history.json', 'summary.json', 'best-predictions.json']:
            left = json.loads((args.focal/'head-fits'/directory/name).read_text())
            right = json.loads((args.small/'fits'/directory/name).read_text())
            checks[directory+'/'+name] = left == right
    if not all(checks.values()):
        raise ValueError('matched geometry control differs; diagnose before comparison')
    focal = json.loads((args.focal/'head-analysis.json').read_text())
    small = json.loads((args.small/'analysis.json').read_text())
    result = dict(geometry_control_exact=checks, focal=focal['arms']['image']['mean_selected_metrics'],
        small=small['arms']['image']['mean_selected_metrics'],
        paired_focal_minus_small=paired(focal['arms']['image']['family_metrics'],
                                       small['arms']['image']['family_metrics']),
        selection_families=7, confirmatory=False, artistic_quality_measured=False,
        face_coverage_certified=False, runtime_latency_measured=False)
    with (args.small/'encoder-comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
