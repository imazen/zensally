import math
import unittest
from dataclasses import replace
from crop_policy import Rect, plan_crops, choose_crop


class CropPolicyTests(unittest.TestCase):
    def test_minimal_keeps_original_when_ratio_does_not_force_crop(self):
        for ratio in (None, (4, 3), (8, 6)):
            p = plan_crops(400, 300, ratio, mode="minimal")
            self.assertEqual(choose_crop(p, []), ("unchanged", Rect(0, 0, 400, 300)))

    def test_all_faces_are_hard_constraints_in_both_modes(self):
        faces = (Rect(185, 2, 195, 16), Rect(205, 280, 215, 299))
        for mode in ("minimal", "maximal"):
            p = plan_crops(400, 300, (9, 16), mode=mode, protected_faces=faces, faces_reviewed=True)
            self.assertTrue(p.candidates)
            for r in p.candidates:
                self.assertTrue(all(r.contains(f) for f in faces))
                self.assertAlmostEqual(r.width / r.height, 9 / 16)
            self.assertEqual(choose_crop(p, range(len(p.candidates)))[0], "crop")

    def test_no_feasible_crop_uses_contain(self):
        faces = (Rect(0, 10, 30, 40), Rect(370, 10, 400, 40))
        for mode in ("minimal", "maximal"):
            p = plan_crops(400, 300, (9, 16), mode=mode, protected_faces=faces, faces_reviewed=True)
            self.assertEqual(choose_crop(p, []), ("contain", p.source))

    def test_empty_detection_is_not_certified_face_free(self):
        p = plan_crops(400, 300, (1, 1), faces_reviewed=False)
        self.assertEqual(choose_crop(p, []), ("contain", p.source))

    def test_adversarial_score_cannot_override_final_face_guard(self):
        face = Rect(0, 0, 10, 10)
        p = plan_crops(400, 300, (1, 1), protected_faces=(face,), faces_reviewed=True)
        unsafe = Rect(100, 0, 400, 300)
        p = replace(p, candidates=(*p.candidates, unsafe))
        kind, r = choose_crop(p, [0.] * (len(p.candidates) - 1) + [1e30])
        self.assertEqual(kind, "crop")
        self.assertTrue(r.contains(face))
        self.assertNotEqual(r, unsafe)

    def test_uncertain_scores_do_not_produce_arbitrary_crop(self):
        p = plan_crops(400, 300, (1, 1), faces_reviewed=True)
        self.assertEqual(choose_crop(p, [math.nan] * len(p.candidates)), ("contain", p.source))

    def test_ratio_and_face_invariants_across_extreme_geometries(self):
        for w, h in ((1, 1), (31, 257), (257, 31), (4096, 127), (127, 4096)):
            face = Rect(w * .45, h * .45, w * .55, h * .55)
            for ratio in ((1, 1), (9, 16), (16, 9), (1, 8), (8, 1)):
                for mode in ("minimal", "maximal"):
                    p = plan_crops(w, h, ratio, mode=mode, protected_faces=(face,), faces_reviewed=True)
                    for r in p.candidates:
                        self.assertTrue(p.source.contains(r) and r.contains(face))
                        self.assertAlmostEqual(r.width / r.height, ratio[0] / ratio[1], places=10)
                    if p.kind == "candidates" and mode == "minimal":
                        self.assertTrue(abs(p.candidates[0].width - w) < 1e-10 or abs(p.candidates[0].height - h) < 1e-10)


if __name__ == "__main__":
    unittest.main()
