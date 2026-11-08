#!/usr/bin/env python3
"""Reconcile unused owned photographic TRAIN candidates into a review-only plan."""
import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'crop-first'))
from corpus import reconcile, rows, save
from photo_focus import category_scope
from artifact_identity import sha


def analysis_witnesses(sources):
    selected, families = [], set()
    for category, count in [('general', 2), ('interiors', 1), ('nature', 2), ('food', 1)]:
        group = sorted((s for s in sources if s['category'] == category),
            key=lambda s: hashlib.sha256(('owned-screen-numerical-witness-v1|'+s['id']).encode()).hexdigest())
        for source in group:
            if source['family_id'] in families:
                continue
            selected.append(source['id'])
            families.add(source['family_id'])
            count -= 1
            if count == 0:
                break
        if count:
            raise ValueError('insufficient independent families for analysis witnesses')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['audit', 'corpus', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.corpus/'protocol.json').read_text())
    for link in protocol['links']:
        path = args.corpus/link['text']
        if sha(path) != link['sha256']:
            raise ValueError('pinned corpus metadata differs: '+str(path))
    exports = {role: rows((args.corpus/f'jxl-{role}.tsv').read_bytes())
               for role in ['train', 'validate', 'test']}
    historical = sum([rows((args.corpus/f'historical-{role}.tsv').read_bytes())
                      for role in ['train', 'validate', 'test']], [])
    records = reconcile(exports, rows((args.corpus/'family-v2.tsv').read_bytes()), historical)
    lookup = {r['id']: r for r in records}
    selected = json.loads((args.corpus/'sources.json').read_text())
    ids = {s['id'] for s in selected}
    families = {s['family_id'] for s in selected}
    keys = ['original_sha256', 'pixels_sha256', 'sha256']
    hashes = {(k, s[k]) for s in selected for k in keys if s.get(k)}
    candidates = json.loads(args.audit.read_text())
    admitted = []
    for entry in candidates:
        source = lookup[entry['id']]
        if any(entry[key] != source[key] for key in ['family_id', 'bucket', 'path',
               'sha256', 'original_sha256', 'pixels_sha256', 'license', 'category']):
            raise ValueError('audit no longer matches pinned source metadata')
        if (source['license'] != 'PD-own' or source['source'] != 'lilith'
                or source['id'] in ids or source['family_id'] in families
                or any((k, source[k]) in hashes for k in keys if source.get(k))):
            continue
        if (source['bucket'] != 'train' or source['historical_eval_family']
                or category_scope(source['category']) != 'photographic_screening_candidate'
                or int(source['width'])*int(source['height']) > 32_000_000):
            raise ValueError('unexpected source in the registered screening cohort')
        admitted.append(dict(source, use='screening', training_eligible=False,
            human_rated=False, face_safety_certified=False, screening_state='pending_visual_review',
            composition_profile='people-commerce-v1'))
    if (len(admitted), len({s['family_id'] for s in admitted})) != (101, 98):
        raise ValueError('registered owned screening cohort differs')
    if len({s['id'] for s in admitted}) != len(admitted):
        raise ValueError('duplicate screening source')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'sources.json', admitted)
    result = dict(sources=len(admitted), families=98,
        categories=dict(Counter(s['category'] for s in admitted)),
        encoded_bytes=sum(int(s['bytes']) for s in admitted),
        declared_rgb8_payload_bytes=sum(int(s['width'])*int(s['height'])*3 for s in admitted),
        audit_sha256=sha(args.audit), old_selection_sha256=sha(args.corpus/'sources.json'),
        source_roles_moved=False, new_training_labels=False, paid_model_api_calls=0)
    protocol = dict(protocol, selected_uses=dict(screening=101),
        admission='unadmitted owned-photo screening only; no gradients or split assignment',
        screening=result)
    save(args.output/'protocol.json', protocol)
    (args.output/'links.jsonl').write_text(''.join(json.dumps(link)+'\n' for link in protocol['links']))
    save(args.output/'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
