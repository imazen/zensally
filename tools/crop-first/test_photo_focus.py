import hashlib
import tempfile
import unittest
from pathlib import Path

import direct_zones
import photo_focus
import review
from photo_audit import audit


class PhotoFocusTests(unittest.TestCase):
    def test_canonical_and_internal_holdouts_never_become_screening_candidates(self):
        def record(identity, **kw):
            return dict(id=identity, family_id=identity, bucket='train',
                        category='people', historical_eval_family=False,
                        width=640, height=480, source='unsplash', **kw)
        records = [record(str(i)) for i in range(5)]
        records[1]['bucket'] = 'validate'
        records[2]['bucket'] = 'test'
        records[3]['historical_eval_family'] = True
        selected = [{**records[4], 'use': 'dev'}]
        _, candidates = audit(records, selected)
        self.assertEqual([r['id'] for r in candidates], ['0'])
        self.assertFalse(candidates[0]['training_eligible'])
        self.assertTrue(candidates[0]['teacher_rights_review_required'])

    def test_documents_and_generated_products_are_not_primary_photo_candidates(self):
        records = [dict(id=c, family_id=c, bucket='train', category=c,
                        historical_eval_family=False, width=640, height=480)
                   for c in sorted(photo_focus.PHOTO_CATEGORIES |
                                   photo_focus.DOCUMENT_CATEGORIES |
                                   photo_focus.GENERATED_CATEGORIES | {'unknown'})]
        _, candidates = audit(records, [])
        self.assertEqual({r['category'] for r in candidates}, photo_focus.PHOTO_CATEGORIES)

    def test_separate_family_with_exact_internal_dev_hash_is_excluded(self):
        source = dict(id='one', family_id='one', bucket='train', category='people',
                      historical_eval_family=False, width=640, height=480,
                      original_sha256='same-original')
        development = {**source, 'id': 'two', 'family_id': 'two', 'use': 'dev'}
        _, candidates = audit([source], [development])
        self.assertEqual(candidates, [])

    def test_profile_is_inherited_by_proposal_and_review_tasks(self):
        source = {'composition_profile': photo_focus.NAME}
        self.assertEqual(photo_focus.inherit(source, {}), source)
        self.assertEqual(photo_focus.inherit({}, {}), {})
        with self.assertRaises(ValueError):
            photo_focus.inherit({'composition_profile': 'typo'}, {})

    def test_request_cache_identity_changes_only_when_photographic_profile_is_selected(self):
        parent = Path.home() / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as name:
            path = Path(name) / 'source.png'
            path.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            task = {'id': 'fixture', 'role': 'train', 'width': 1200, 'height': 800,
                    'image': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                    'tasks': []}
            for builder, legacy, sentence in [
                (direct_zones.build, direct_zones.PROMPT, 'For screenshots/documents preserve meaningful text.'),
                (review.build, review.PROMPT, 'Preserve meaningful text in documents/screenshots.'),
            ]:
                original = builder(task)
                self.assertEqual(original['input'][0]['content'], legacy)
                changed = builder({**task, 'composition_profile': photo_focus.NAME})
                text = changed['input'][0]['content']
                self.assertNotIn(sentence, text)
                self.assertIn('People are the priority', text)
                self.assertIn('pets and other animals', text)
                self.assertIn('ecommerce', text)
                self.assertNotEqual(direct_zones.sha(direct_zones.canonical(original)),
                                    direct_zones.sha(direct_zones.canonical(changed)))
                self.assertEqual(builder(task), original)
                with self.assertRaises(ValueError):
                    builder({**task, 'composition_profile': 'typo'})


if __name__ == '__main__':
    unittest.main()
