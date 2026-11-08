#!/usr/bin/env python3
"""Assign provider components once, before new composition labels or fits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['previews', 'relationships', 'detectors', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    manifest = args.previews/'manifest.json'
    components_path = args.relationships/'components.json'
    summary_path = args.relationships/'summary.json'
    rows = json.loads(manifest.read_text())['rows']
    components = json.loads(components_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (sha(manifest) != '9d7f1f44c092084391b31a172a8253536686b2b0ec3fca9f5e1250164e4b6cda'
            or summary['sources'] != 144 or summary['pairs'] != 10296 or not summary['calibration_exact']
            or len(components) != 124 or sorted(i for c in components for i in c) != sorted(r['id'] for r in rows)):
        raise ValueError('registered component/pixel screen differs')
    count_doc = json.loads((args.detectors/'coverage-analysis.json').read_text())
    counts = count_doc['source_counts']
    bindings, records, photographer_roles = {}, [], defaultdict(set)
    for component in sorted(components):
        ids = sorted(component)
        digest = hashlib.sha256(('zensally-unsplash-role-v1\0seed31\0'+ids[0]).encode()).digest()
        slot = int.from_bytes(digest[:8], 'big') % 10000
        role = 'fit' if slot < 7000 else 'dev' if slot < 8500 else 'test'
        family = 'unsplash-lineage:'+hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        for identifier in ids:
            if identifier in bindings:
                raise ValueError('source belongs to multiple components')
            bindings[identifier] = dict(family_id=family, role=role, role_anchor_id=ids[0], hash_slot=slot)
    for row in rows:
        source = row['source']
        if source['canonical_overlap'] or source['training_eligible'] or source['family_role'] is not None:
            raise ValueError('canonical conflict or prior role needs explicit quarantine')
        binding = bindings[row['id']]
        photographer_roles[source['provisional_group']].add(binding['role'])
        exposure = ['owned_decode_readback', 'local_object_detector_screen', 'owned_relationship_fingerprint']
        if source['provider_id'] in ['73Fjvy6Z9w0', 'V-45oWTe9k4']:
            exposure.append('assistant_visual_detector_review')
        records.append(dict(id=row['id'], **binding, source=source,
            native_size=row['native_size'], working_size=[row['width'], row['height']],
            rgb=row['rgb'], rgb_sha256=row['rgb_sha256'], png=row['png'], png_sha256=row['png_sha256'],
            prior_exposure=exposure, crop_teacher_scores_opened_at_role_freeze=False,
            detected_person_boxes=counts[row['id']]['int8-efficientdet_lite0'],
            training_eligible=False, human_rated=False, face_coverage_certified=False))
    if any(len(roles) != 1 for roles in photographer_roles.values()):
        raise ValueError('photographer crosses roles')
    result = dict(schema='zensally-unsplash-role-registry-v1', sources=records,
        preview_manifest_sha256=sha(manifest), components_sha256=sha(components_path),
        relationship_summary_sha256=sha(summary_path), source_roles_changed=False,
        existing_canonical_roles_changed=False, future_alias_policy='inherit fixed registry; conflicts quarantine',
        new_roles_assigned=True, training_labels_admitted=False,
        canonical_holdout_pixels_opened=False, diagnostic_maps_saved=False, paid_model_api_calls=0)
    summary = dict(sources=dict(Counter(r['role'] for r in records)),
        components=dict(Counter(bindings[c[0]]['role'] for c in components)),
        detector_person_positive_sources=dict(Counter(r['role'] for r in records if r['detected_person_boxes'])),
        prior_assistant_detector_reviews=[dict(id=r['id'], role=r['role']) for r in records
                                        if 'assistant_visual_detector_review' in r['prior_exposure']],
        seed_rerolled=False, human_content_coverage_measured=False, training_labels_admitted=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'registry.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(summary, registry_sha256=sha(args.output/'registry.json'))))


if __name__ == '__main__':
    main()
