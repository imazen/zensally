#!/usr/bin/env python3
"""Independent screenshot pixels must show background outside the crop clip."""
import argparse
import json
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('browser_report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    browser = json.loads(args.browser_report.read_text())
    screenshot = args.screenshot or Path(browser['witness_screenshot'])
    samples = browser['gutter_samples']
    if len(samples) < 4:
        raise ValueError('two tall crop witnesses with both gutters required')
    checks = []
    with Image.open(screenshot) as image:
        rgb = image.convert('RGB')
        for sample in samples:
            x, y = sample['x'], sample['y']
            if not (1 <= x < rgb.width-1 and 1 <= y < rgb.height-1):
                raise ValueError('gutter sample outside screenshot')
            pixels = [list(rgb.getpixel((x+dx, y+dy))) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            checks.append(dict(**sample, actual_pixels=pixels,
                passed=all(pixel == sample['expected_rgb'] for pixel in pixels)))
    report = dict(screenshot=str(screenshot), checks=checks,
                  failed_samples=sum(not c['passed'] for c in checks),
                  purpose='browser crop clipping only; no training/metric diagnostic maps')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(samples=len(checks), failed_samples=report['failed_samples'])))
    if report['failed_samples']:
        raise ValueError('surrounding image pixels escape the intended SVG crop')


if __name__ == '__main__':
    main()
