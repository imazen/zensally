#!/usr/bin/env python3
"""Compare existing photographic candidate pixels without changing labels."""
import argparse
import json
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['rendered', 'cohort', 'photo_fits', 'mixed_fits', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--asset-base', required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    members = [m for m in cohort['members'] if m['use'] == 'dev']
    if len(members) != 5 or cohort['name'] != 'existing-photo-preferences-v1':
        raise ValueError('registered five-source development cohort required')
    paths = {
        'Frozen FocalNet rank': args.photo_fits/'visual-seed31/predictions-epoch-000.json',
        'Mixed-corpus fit': args.mixed_fits/'visual-seed31/best-predictions.json',
        'Photo-only fit': args.photo_fits/'visual-seed31/best-predictions.json',
        'Photo geometry control': args.photo_fits/'geometry-seed31/best-predictions.json',
    }
    predictions = {name: {p['task_id']: p for p in json.loads(path.read_text())}
                   for name, path in paths.items()}
    rows = []
    for member in members:
        root = args.rendered/member['id']
        packet = json.loads((root/'packet.json').read_text())
        if any(packet[key] != member[key] for key in ['id', 'family_id', 'use']):
            raise ValueError('gallery source identity differs')
        if not (root/'source.png').is_file():
            raise ValueError('required source pixels missing')
        row = {key: packet[key] for key in ['id', 'family_id', 'use', 'sha256',
                                           'canonical_url', 'license', 'provenance']}
        row['image'] = args.asset_base.rstrip('/')+'/'+member['id']+'/source.png'
        row['tasks'] = []
        for task in packet['tasks']:
            choices = []
            for name, lookup in predictions.items():
                p = lookup[task['task_id']]
                if p['source_id'] != member['id']:
                    raise ValueError('gallery prediction source differs')
                candidate = next(c for c in task['candidates'] if c['id'] == p['raw_rank_candidate_id'])
                if not (root/candidate['file']).is_file():
                    raise ValueError('required rendered candidate pixels missing')
                choices.append(dict(model=name, candidate=candidate['id'],
                    image=args.asset_base.rstrip('/')+'/'+member['id']+'/'+candidate['file'],
                    rect=candidate['xywh'], action=p['research_action'],
                    weak_acceptable=p['rank_teacher_acceptable'], judged=p['judged']))
            row['tasks'].append(dict(id=task['task_id'], ratio=task['ratio'],
                                      mode=task['mode'], choices=choices))
        rows.append(row)
    document = dict(study='photo-preference-heads-20261108-v1', sources=rows,
                    predictions={name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
                    cohort_sha256=sha(args.cohort), seed=31,
                    face_coverage_certified=False, human_rated=False)
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('preference_gallery.html').read_text()
    (args.output/'index.html').write_text(template.replace('/*DOCUMENT*/null', json.dumps(document).replace('<', '\\u003c')))
    (args.output/'selections.json').write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows), tasks=sum(len(r['tasks']) for r in rows),
                          new_pixels_generated=False, labels_changed=False)))


if __name__ == '__main__':
    main()
