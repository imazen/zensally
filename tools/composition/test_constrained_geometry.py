"""Independent exhaustive small-image oracle for constrained candidate geometry."""
import math
import unittest

from constrained_bank import candidates


class ConstrainedGeometry(unittest.TestCase):
    def test_largest_feasibility_matches_exhaustive_origins(self):
        for width in range(2, 10):
            for height in range(2, 10):
                regions = [[], [[0, 0, 1, 1]], [[width-1, height-1, width, height]],
                    [[0, 0, 1, 1], [width-1, height-1, width, height]],
                    [[width//3, height//3, width-width//3, height-height//3]]]
                for p,q in [(1, 1), (3, 1), (1, 3), (2, 3), (6, 4)]:
                    divisor = math.gcd(p, q)
                    p,q = p//divisor, q//divisor
                    maximum = min(width//p, height//q)
                    for required in regions:
                        possible = any(all(x <= r[0] and y <= r[1] and x+maximum*p >= r[2]
                            and y+maximum*q >= r[3] for r in required)
                            for y in range(height-maximum*q+1) for x in range(width-maximum*p+1)) if maximum else False
                        result, mw, mh = candidates(width, height, (p, q), required)
                        self.assertEqual(bool(result), possible, (width, height, p, q, required))
                        self.assertLessEqual(len(result), 128)
                        for l,t,r,b in result:
                            self.assertTrue(0 <= l < r <= width and 0 <= t < b <= height)
                            self.assertEqual((r-l)*q, (b-t)*p)
                            self.assertTrue(all(l <= a and t <= c and r >= d and b >= e for a,c,d,e in required))
                        self.assertEqual((mw, mh), (maximum*p, maximum*q))


if __name__ == '__main__':
    unittest.main()
