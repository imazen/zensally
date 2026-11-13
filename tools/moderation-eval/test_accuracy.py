"""Check threshold ties, split leakage, and confusion counts without model data."""
import unittest

import accuracy


class CalibrationTests(unittest.TestCase):
    def test_tied_negatives_respect_budget(self):
        rows = [dict(label=False, score=x) for x in [0.9, 0.8, 0.8, 0.1]]
        cut = accuracy.threshold(rows, 0.5)
        self.assertEqual(accuracy.metrics(rows, cut)["fp"], 1)
        self.assertGreater(cut, 0.8)

    def test_zero_false_positive_budget_and_confusion_counts(self):
        rows = [dict(label=False, score=0.6), dict(label=False, score=0.2),
                dict(label=True, score=0.9), dict(label=True, score=0.5)]
        result = accuracy.metrics(rows, accuracy.threshold(rows, 0))
        self.assertEqual([result[x] for x in ("tp", "fp", "fn", "tn")], [1, 0, 1, 2])
        self.assertEqual(result["recall"], 0.5)
        self.assertGreater(result["false_positive_rate_wilson95"][1], 0)

    def test_source_family_cannot_cross_splits(self):
        rows = []
        for i, (split, label) in enumerate([("calibration", False), ("calibration", True), ("eval", False), ("eval", True)]):
            rows.append(dict(path="unused", sha256=str(i), family="same-source", content_type="photo",
                             source_url="https://example.org/source", license="test", split=split, label=label))
        with self.assertRaisesRegex(ValueError, "family crosses"):
            accuracy.validate(dict(policy="test", annotation_provenance="human", items=rows))


if __name__ == "__main__":
    unittest.main()
