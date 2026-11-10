#!/usr/bin/env python3
"""Build the bounded purpose comparison from retained native rectangles."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    doc = json.loads((args.evaluation/'comparison.json').read_text())
    if len(doc['sources']) != 6 or doc['summary']['requests'] != 330 or doc['summary']['selected_region_violations']:
        raise ValueError('complete registered comparison required')
    template = Path(__file__).with_name('unsplash_purpose_gallery.html').read_text()
    html = template.replace('/*CLIP*/', Path(__file__).with_name('crop_clip.js').read_text())
    html = html.replace('/*DOCUMENT*/null', json.dumps(doc).replace('</', '<\\/'))
    with (args.evaluation/'index.html').open('x') as stream:
        stream.write(html)
    print(json.dumps(dict(sources=6, requests=330, automatic_training_admission=False)))


if __name__ == '__main__':
    main()
