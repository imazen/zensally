#!/usr/bin/env python3
"""Paginated review of unadmitted real photos, crops and detector disagreements."""
import argparse
import json
import math
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--asset-base', required=True)
    args = parser.parse_args()
    document = json.loads((args.root/'inputs.json').read_text())
    observations = {r['id']: r['observation'] for r in json.loads(
        (args.root/'witness-observations.json').read_text())['sources']}
    objects = {}
    for name, directory in [('EfficientDet', 'objects-efficientdet'), ('SSD', 'objects-ssd')]:
        result = json.loads((args.root/directory/'results.json').read_text())
        if [r['id'] for r in result['rows']] != [s['id'] for s in document['images']]:
            raise ValueError('object model/source list differs')
        objects[name] = {r['id']: r['result']['detections'] for r in result['rows']}
    rows = []
    for source in document['images']:
        bank_path = args.root/'crop-proposals'/(source['id']+'.json')
        bank = json.loads(bank_path.read_text())
        if bank['use'] != 'screening' or (bank['width'], bank['height']) != (source['native_width'], source['native_height']):
            raise ValueError('gallery bank/source identity differs')
        if any(len(pose) != 33 or any(not math.isfinite(point[key]) for point in pose
                for key in ['x', 'y', 'z']) for pose in bank['pose']):
            raise ValueError('invalid pose coordinates in gallery')
        people = {}
        for name, lookup in objects.items():
            boxes = []
            for detection in lookup[source['id']]:
                scores = [c['score'] for c in detection['categories'] if c['name'] == 'person']
                if not scores:
                    continue
                x, y, w, h = detection['xywh']
                box = [max(0, x/source['width']), max(0, y/source['height']),
                       min(1, (x+w)/source['width']), min(1, (y+h)/source['height'])]
                if not (box[0] < box[2] and box[1] < box[3]):
                    raise ValueError('person rectangle outside source')
                boxes.append(dict(box=box, score=max(scores)))
            people[name] = boxes
        tasks = {}
        for task in bank['tasks']:
            p, q = map(int, task['ratio'].split(':'))
            choices = {}
            for selection in task['selected']:
                index = selection['detected_head_guarded_candidate']
                value = None if index is None else task['candidates'][index]
                if value:
                    l, t, r, b = value['pixel_box']
                    if ((r-l)*q != (b-t)*p or not (0 <= l < r <= bank['width'] and 0 <= t < b <= bank['height'])):
                        raise ValueError('gallery crop geometry invalid')
                choices[selection['mode']] = None if value is None else dict(
                    box=value['box'], pixel_box=value['pixel_box'],
                    retained_importance=value['retained_importance'],
                    minimal_fallback=selection['minimal_fallback'])
            tasks[task['ratio']] = choices
        rows.append(dict(id=bank['id'], family_id=bank['family_id'], width=bank['width'], height=bank['height'],
            role='screening', heads=bank['heads'], pose=bank['pose'], people=people, tasks=tasks,
            image=args.asset_base.rstrip('/')+'/'+bank['id']+'.png',
            source_url=bank['canonical_url'], license=bank['license'], category=bank['provenance']['category'],
            source_sha256=source['source_jxl_sha256'], bank_sha256=sha(bank_path),
            observation=observations.get(bank['id']), face_coverage_certified=False, training_eligible=False))
    if len(rows) != 101:
        raise ValueError('complete registered gallery requires101 photos')
    data = dict(study='owned-photo-screen-20261108-v1', sources=rows,
                render_contract='explicit-source-coordinate-clip-v2',
                inputs_sha256=sha(args.root/'inputs.json'), face_coverage_certified=False,
                training_admission_automatic=False)
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('owned_screen_gallery.html').read_text()
    template = template.replace('/*CLIP_HELPER*/', Path(__file__).with_name('crop_clip.js').read_text())
    (args.output/'index.html').write_text(template.replace('/*DOCUMENT*/null', json.dumps(data).replace('<', '\\u003c')))
    (args.output/'selections.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows), per_page=12, new_pixels_generated=False)))


if __name__ == '__main__':
    main()
