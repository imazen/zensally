import unittest
from pose_head_bounds import head_bounds


class PoseHeadTest(unittest.TestCase):
    def test_shoulders_can_bound_head_and_round_outward(self):
        points = [dict(x=40+i*2, y=40+i, visibility=1., presence=1.) for i in range(11)]
        points += [dict(x=10, y=100, visibility=1., presence=1.),
                   dict(x=90, y=100, visibility=1., presence=1.)]
        points += [points[0].copy() for _ in range(20)]
        self.assertEqual(head_bounds(points, 100, 100)[0], [24, 0, 76, 63])
        points[8]['presence'] = .49
        self.assertIsNone(head_bounds(points, 100, 100)[0])


if __name__ == '__main__':
    unittest.main()
