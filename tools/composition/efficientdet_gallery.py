#!/usr/bin/env python3
"""Compare local object-detector coordinates on the same fixed owned photos."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--asset-base', required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root/'manifest.json').read_text())
    models = []
    for model in manifest['models']:
        result = json.loads((args.root/'full'/model['id']/'results.json').read_text())
        if [r['id'] for r in result['rows']] != [r['id'] for r in manifest['images']]:
            raise ValueError('fixed input/result ordering differs')
        models.append(dict(id=model['id'], bytes=model['bytes'], sha256=model['sha256'],
                           detections={r['id']:r['result']['detections'] for r in result['rows']}))
    doc = dict(models=models, sources=[dict(id=r['id'], width=r['width'], height=r['height'],
        image=args.asset_base.rstrip('/')+'/'+r['id']+'.png', rgb_sha256=r['rgb_sha256']) for r in manifest['images']],
        manifest_sha256=sha(args.root/'manifest.json'), face_coverage_certified=False,
        training_labels_admitted=False)
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('efficientdet_gallery.html').read_text()
    template = template.replace('/*CLIP_HELPER*/', Path(__file__).with_name('crop_clip.js').read_text())
    (args.output/'index.html').write_text(template.replace('/*DOCUMENT*/null', json.dumps(doc).replace('<', '\\u003c')))
    print(json.dumps(dict(sources=len(doc['sources']), models=len(models), new_pixels_generated=False)))


if __name__ == '__main__':
    main()
