import random
import unittest

from roi_crop_geometry import place, solve, rank_regions
from subject_policy import face_relation


class RoiGeometry(unittest.TestCase):
    def test_origin_matches_independent_exhaustive_search(self):
        rng = random.Random(211)
        for _ in range(300):
            w, h = rng.randrange(3, 15), rng.randrange(3, 15)
            def box():
                x, y = rng.randrange(w), rng.randrange(h)
                return [x, y, rng.randrange(x+1, w+1), rng.randrange(y+1, h+1)]
            roi, faces = box(), [box() for _ in range(rng.randrange(5))]
            cw, ch = rng.randrange(1, w+1), rng.randrange(1, h+1)
            anchor = [rng.randrange(w), rng.randrange(h)]
            options = []
            for y in range(h-ch+1):
                for x in range(w-cw+1):
                    crop = [x, y, x+cw, y+ch]
                    if (x <= roi[0] and y <= roi[1] and x+cw >= roi[2] and y+ch >= roi[3]
                            and all(face_relation(crop, f) != 'cut' for f in faces)):
                        options.append(((((x-anchor[0])*h)**2+((y-anchor[1])*w)**2, y, x), crop))
            self.assertEqual(place(w, h, cw, ch, roi, faces, anchor), min(options)[1] if options else None)

    def test_largest_area_and_whole_interval(self):
        rng = random.Random(811)
        for _ in range(100):
            w, h = rng.randrange(4, 15), rng.randrange(4, 15)
            x, y = rng.randrange(w), rng.randrange(h)
            roi = [x, y, min(x+2, w), min(y+2, h)]
            faces = [roi, [0, 0, min(2, w), min(2, h)]]
            p, q = rng.choice([(1, 1), (3, 1), (1, 3)])
            result = solve(w, h, f'{p}:{q}', roi, faces, padding_percent=0)
            feasible = []
            for k in range(1, min(w//p, h//q)+1):
                if any(xx <= roi[0] and yy <= roi[1] and xx+k*p >= roi[2] and yy+k*q >= roi[3]
                       and all(face_relation([xx, yy, xx+k*p, yy+k*q], f) != 'cut' for f in faces)
                       for yy in range(h-k*q+1) for xx in range(w-k*p+1)):
                    feasible.append(k)
            if not feasible:
                self.assertIsNone(result['outer'])
                continue
            outer, inner = result['outer'], result['inner']
            self.assertEqual(outer[2]-outer[0], max(feasible)*p)
            if inner:
                self.assertTrue(outer[0] <= inner[0] < inner[2] <= outer[2])
                self.assertTrue(outer[1] <= inner[1] < inner[3] <= outer[3])
                for face in faces:
                    self.assertTrue(face_relation(inner, face) == 'contained' or face_relation(outer, face) == 'excluded')

    def test_selection_can_exclude_another_person_and_has_no_center_prior(self):
        faces = [[1, 1, 3, 3], [16, 1, 18, 3]]
        left = solve(20, 8, '1:1', faces[0], faces)
        right = solve(20, 8, '1:1', faces[1], faces)
        self.assertLess(left['outer'][0], right['outer'][0])
        self.assertEqual(left['excluded_faces'], [1])
        self.assertEqual(right['excluded_faces'], [0])
        self.assertEqual(solve(1, 1, '1:1', [0, 0, 1, 1], [])['status'], 'single-frame-no-distinct-tight-crop')

    def test_invalid_and_impossible(self):
        for args in [(0, 4, '1:1', [0, 0, 1, 1], []), (4, 4, '0:1', [0, 0, 1, 1], []),
                     (4, 4, '1:1', [0, 0, 5, 1], [])]:
            with self.assertRaises(ValueError): solve(*args)
        self.assertIsNone(solve(8, 8, '3:1', [0, 0, 8, 8], [])['outer'])
        with self.assertRaises(ValueError): rank_regions([dict(box=[0, 0, 1, 1], kind='x', confidence=float('nan'))], 4, 4)


if __name__ == '__main__': unittest.main()
