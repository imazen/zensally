import importlib.util
import unittest
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('feasibility', Path(__file__).resolve().parents[1] / 'tools/composition/audit_crop_feasibility.py')
feasibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feasibility)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/composition'))
from constrained_bank import candidates


class FeasibilityTests(unittest.TestCase):
    def test_exact_ratio_offset_and_separated_heads(self):
        self.assertEqual(feasibility.interval(100, 100, (1, 3), [[41, 20, 59, 80]], 33), [26, 0, 59, 99])
        self.assertIsNone(feasibility.interval(100, 100, (1, 3), [[0, 0, 10, 10], [90, 90, 100, 100]], 33))
        self.assertIsNone(feasibility.interval(3, 3, (6, 1), [], 0))

    def test_bank_recovers_offset_and_all_candidates_have_integer_guards(self):
        boxes, _, _ = candidates(100, 100, (1, 3), [[41, 20, 59, 80]])
        self.assertTrue(boxes)
        self.assertLessEqual(len(boxes), 128)
        for left, top, right, bottom in boxes:
            self.assertEqual((right-left)*3, bottom-top)
            self.assertTrue(0 <= left <= 41 < 59 <= right <= 100)
            self.assertTrue(0 <= top <= 20 < 80 <= bottom <= 100)
        # Exhaustive independent integer oracle on a small separated-region case.
        heads = [[1, 2, 3, 4], [5, 5, 7, 6]]
        for p, q in [(1, 3), (3, 1), (1, 1)]:
            exists = any(all(x <= h[0] and y <= h[1] and x+k*p >= h[2] and y+k*q >= h[3] for h in heads)
                for k in range(1, 9) for x in range(9) for y in range(9)
                if x+k*p <= 8 and y+k*q <= 8)
            generated, _, _ = candidates(8, 8, (p, q), heads)
            self.assertEqual(bool(generated), exists)


if __name__ == '__main__':
    unittest.main()
