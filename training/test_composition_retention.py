import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/composition'))
from region_retention import evaluate, retained_fraction


class RetentionTests(unittest.TestCase):
    def test_region_area_and_integer_authority(self):
        self.assertEqual(retained_fraction([0,0,2,2], [1,0,3,2]), .5)
        self.assertEqual(retained_fraction([0,0,1,1], [2,2,3,3]), 0)
        annotation = dict(id='x', width=100, height=100,
            regions=[dict(kind='product', description='object', box_percent=[10,10,50,50])])
        bank = dict(id='x', width=100, height=100, tasks=[dict(ratio='1:1',
            candidates=[dict(pixel_box=[10,10,50,50],box=[.10000001,.1,.5,.5])],
            selected=[dict(mode='minimal',detected_head_guarded_candidate=0),
                      dict(mode='maximal',detected_head_guarded_candidate=None)])])
        rows = evaluate(annotation, bank)
        self.assertTrue(all(r['all_regions_contained'] for r in rows))
        self.assertEqual(rows[1]['action'], 'contain')
        self.assertIsNone(rows[0]['aesthetic_grade'])
        bank['tasks'][0]['candidates'][0]['pixel_box']=[10,10,30,50]
        self.assertFalse(evaluate(annotation,bank)[0]['all_regions_contained'])

    def test_invalid_geometry_is_an_error(self):
        for invalid in [[0,0,0,1], [float('nan'),0,1,1]]:
            with self.assertRaises(ValueError): retained_fraction(invalid,[0,0,1,1])


if __name__ == '__main__':
    unittest.main()
