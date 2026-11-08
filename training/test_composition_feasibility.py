import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('feasibility', Path(__file__).resolve().parents[1] / 'tools/composition/audit_crop_feasibility.py')
feasibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feasibility)


class FeasibilityTests(unittest.TestCase):
    def test_exact_ratio_offset_and_separated_heads(self):
        self.assertEqual(feasibility.interval(100, 100, (1, 3), [[41, 20, 59, 80]], 33), [26, 0, 59, 99])
        self.assertIsNone(feasibility.interval(100, 100, (1, 3), [[0, 0, 10, 10], [90, 90, 100, 100]], 33))
        self.assertIsNone(feasibility.interval(3, 3, (6, 1), [], 0))


if __name__ == '__main__':
    unittest.main()
