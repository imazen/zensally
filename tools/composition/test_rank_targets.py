import unittest
import numpy as np
from rank_targets import targets


class RankTargets(unittest.TestCase):
    def test_singleton_constant_and_absent_queries_stay_unknown(self):
        values = np.ones(128, np.float32)
        for selected in [[], [7], [1, 5, 9]]:
            mask, available, target, pairs, valid = targets(values, selected)
            self.assertFalse(available)
            self.assertEqual(int(mask.sum()), len(selected))
            self.assertFalse(valid.any())
            self.assertEqual(pairs.shape, (32, 2))
            self.assertTrue((target == 0).all())

    def test_preferences_point_to_higher_eligible_scores(self):
        values = np.arange(128, dtype=np.float32)
        selected = [1, 4, 9, 17, 31, 54, 88, 100]
        mask, available, target, pairs, valid = targets(values, selected)
        self.assertTrue(available)
        self.assertAlmostEqual(float(target[mask].mean()), 0, places=6)
        self.assertAlmostEqual(float(target[mask].std()), 1, places=6)
        self.assertTrue((target[~mask] == 0).all())
        self.assertGreater(int(valid.sum()), 0)
        for high, low in pairs[valid]:
            self.assertIn(int(high), selected)
            self.assertIn(int(low), selected)
            self.assertGreaterEqual(float(target[high]-target[low]), .25)
            self.assertGreater(float(values[high]), float(values[low]))


if __name__ == '__main__':
    unittest.main()
