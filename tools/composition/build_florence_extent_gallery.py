#!/usr/bin/env python3
"""Review source-bound weak targets and selected extent heads in native geometry."""
import argparse
import json
from pathlib import Path
import shutil

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    plan = json.loads((args.root/'source-plan.json').read_text())
    admission = json.loads((args.root/'extent-cache/admission.json').read_text())
    targets = {s['id']:s['target'] for s in admission['sources']}
    entities = {s['id']:s['entities'] for s in admission['teacher_identities']}
    models = {}
    for name,directory in [('small',args.root/'extent-fits'),('focal',args.root/'focalnet/extent-fits')]:
        # Display every seed; do not pick a winner after inspecting the images.
        for seed in [31,137,911]:
            models[name+'-'+str(seed)] = {s['id']:s['box'] for s in json.loads(
                (directory/('image-seed'+str(seed))/'best-predictions.json').read_text())}
    gallery = args.root/'review'
    gallery.mkdir(exist_ok=False)
    rows = []
    for source in plan['sources']:
        original = Path(source['runtime_source']['display'])
        image = source['id']+'.png'
        shutil.copy2(original,gallery/image)
        w,h = source['native_size']
        rows.append(dict(id=source['id'], role=source['role'], family_id=source['family_id'],
            query=source['query'], size=[w,h], image=image, image_sha256=sha(gallery/image),
            target=targets[source['id']], entities=entities[source['id']],
            models={k:v.get(source['id']) for k,v in models.items()}))
    data = dict(sources=rows, human_rated=False, automatic_training_admission=False)
    html = Path(__file__).with_name('florence_extent_gallery.html').read_text()
    html = html.replace('/*DATA*/null',json.dumps(data).replace('</','<\\/'))
    with (gallery/'index.html').open('x') as stream:
        stream.write(html)
    print(json.dumps(dict(sources=len(rows), development_predictions=8*6)))


if __name__ == '__main__':
    main()
