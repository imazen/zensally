import unittest

from roi_geometry import person_views, project_face


class RoiGeometryTest(unittest.TestCase):
    def test_outward_rounding_context_and_upper_window(self):
        self.assertEqual(person_views([10, 20, 20, 40], [100, 100], [1000, 1000]),
            [('person', [80, 160, 320, 640]), ('upper', [80, 160, 320, 420])])
        self.assertEqual(person_views([0, 0, 1, 1], [3, 3], [10, 10]),
            [('person', [0, 0, 5, 5]), ('upper', [0, 0, 5, 3])])

    def test_translate_scale_and_partial_visible_box(self):
        result = project_face([-10, 20, 30, 40], [0, 160, 240, 640], [120, 240], [1000, 1000])
        self.assertEqual(result['raw_native_box'], [-20, 200, 40, 280])
        self.assertEqual(result['visible_native_box'], [0, 200, 40, 280])
        self.assertIsNone(project_face([-40, 0, 20, 10], [0, 0, 100, 100],
                                      [100, 100], [1000, 1000])['visible_native_box'])


if __name__ == '__main__':
    unittest.main()
