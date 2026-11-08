#!/usr/bin/env python3
"""Audit frozen corpus metadata for the photographic successor; no image reads."""
import argparse
from collections import Counter
import json
from pathlib import Path

from corpus import reconcile, rows, save
from photo_focus import NAME, category_scope


def audit(records, selected):
    # Previously assigned internal development families never migrate into Fit.
    dev_families = {s['family_id'] for s in selected if s['use'] == 'dev'}
    hash_keys = ['original_sha256', 'pixels_sha256', 'sha256']
    dev_hashes = {(k, s[k]) for s in selected if s['use'] == 'dev'
                  for k in hash_keys if s.get(k)}
    candidates = []
    for r in records:
        scope = category_scope(r['category'])
        if (r['bucket'] != 'train' or r['historical_eval_family'] or
                r['family_id'] in dev_families or
                any((k, r[k]) in dev_hashes for k in hash_keys if r.get(k)) or
                int(r['width']) * int(r['height']) > 32_000_000 or
                min(int(r['width']), int(r['height'])) < 128 or
                scope != 'photographic_screening_candidate'):
            continue
        candidates.append({**r, 'composition_profile': NAME,
                           'screening_state': 'pending_visual_review',
                           'training_eligible': False,
                           'teacher_rights_review_required': r.get('source') == 'unsplash'})
    fitting = [r for r in selected if r['use'] == 'fit']
    summary = {
        'profile': NAME,
        'metadata_only': True,
        'existing_fit_sources': len(fitting),
        'existing_fit_by_scope': dict(Counter(category_scope(r['category']) for r in fitting)),
        'existing_fit_by_category': dict(Counter(r['category'] for r in fitting)),
        'new_TRAIN_screening_candidates_by_category': dict(Counter(r['category'] for r in candidates)),
        'new_TRAIN_screening_families': len({r['family_id'] for r in candidates}),
        'people_category_is_not_a_people_detector': True,
        'all_candidates_need_visual_screening_and_rights_admission': True,
        'existing_dev_validate_test_assignments_unchanged': True,
    }
    return summary, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('corpus', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    exports = {role: rows((a.corpus / f'jxl-{role}.tsv').read_bytes())
               for role in ['train', 'validate', 'test']}
    historical = sum([rows((a.corpus / f'historical-{role}.tsv').read_bytes())
                      for role in ['train', 'validate', 'test']], [])
    records = reconcile(exports, rows((a.corpus / 'family-v2.tsv').read_bytes()), historical)
    selected = json.loads((a.corpus / 'sources.json').read_text())
    summary, candidates = audit(records, selected)
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output / 'audit.json', summary)
    save(a.output / 'TRAIN-screening-candidates.json', candidates)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
