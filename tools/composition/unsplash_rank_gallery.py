#!/usr/bin/env python3
"""Review all development people witnesses plus three fixed non-person photos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'registry', 'fits', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if (sha(args.cache) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'
            or sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6'):
        raise ValueError('fixed development review inputs differ')
    sources = json.loads(args.registry.read_text())['sources']
    people = [s for s in sources if s['role'] == 'dev' and s['detected_person_boxes']]
    others = sorted([s for s in sources if s['role'] == 'dev' and not s['detected_person_boxes']],
                    key=lambda s:hashlib.sha256(('review-source-v1\0'+s['id']).encode()).digest())
    selected = people+others[:3]
    if len(people) != 3 or len(selected) != 6:
        raise ValueError('all3 development people plus3 fixed other sources required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(
        selected_ids=[s['id'] for s in selected], selection='all detector-positive development people plus3 SHA256-ordered others',
        training_admission=False, face_coverage_certified=False, sources_chosen_before_rendering=True), indent=2)+'\n')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    predictions = {}
    for arm in ['trainable-image', 'geometry-only']:
        root = args.fits/(arm+'-seed911')
        report = json.loads((root/'summary.json').read_text())
        if sha(root/'best.pt') != report['best_weights_sha256']:
            raise ValueError('selected review model differs')
        predictions[arm] = {(r['id'], r['query']):r for r in json.loads((root/'best-predictions.json').read_text())}
    rows = []
    for source in selected:
        i = next(i for i,s in enumerate(cache['sources']) if s['id'] == source['id'])
        image = Path(source['png'])
        if sha(image) != source['png_sha256']:
            raise ValueError('owned source preview differs')
        name = source['source']['provider_id']+'.png'
        shutil.copy2(image, args.output/name)
        tasks = []
        for q in range(22):
            learned = predictions['trainable-image'][(source['id'], q)]
            geometry = predictions['geometry-only'][(source['id'], q)]
            bank = cache['native_boxes'][i][q//2]
            choices = dict(teacher=learned['teacher'], student=learned['selected'], geometry=geometry['selected'])
            boxes = {key:bank[j] for key,j in choices.items()}
            width, height = source['native_size']
            p, ratio_q = map(int, cache['ratios'][q//2].split(':'))
            for l,t,r,b in boxes.values():
                if not (0 <= l < r <= width and 0 <= t < b <= height and (r-l)*ratio_q == (b-t)*p):
                    raise ValueError('review integer/aspect geometry differs')
            tasks.append(dict(query=q, ratio=cache['ratios'][q//2], mode='minimal' if q % 2 == 0 else 'maximal',
                              boxes=boxes, choices=choices, teacher_query_informative=learned['query_valid']))
        rows.append(dict(id=source['id'], family_id=source['family_id'], role='dev',
            image=name, native_size=source['native_size'], source_url=source['source']['source_url'],
            photographer=source['source']['photographer'], detector_person_boxes=source['detected_person_boxes'], tasks=tasks))
    doc = dict(sources=rows, ratios=cache['ratios'], human_artistic_quality_measured=False,
               face_coverage_certified=False, model_promoted=False, training_admission=False)
    (args.output/'selections.json').write_text(json.dumps(doc, indent=2)+'\n')
    template = Path(__file__).with_suffix('.html').read_text()
    clip = Path(__file__).with_name('crop_clip.js').read_text()
    (args.output/'index.html').write_text(template.replace('/*CLIP*/', clip).replace('/*DOCUMENT*/null', json.dumps(doc).replace('</', '<\\/')))
    print(json.dumps(dict(sources=6, source_requests=132, comparison_crops=396,
                          training_admission=False, face_coverage_certified=False)))


if __name__ == '__main__':
    main()
