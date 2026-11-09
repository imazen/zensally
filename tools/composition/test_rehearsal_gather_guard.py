from pathlib import Path
import tempfile
import unittest
from gather_grid import lower_student_grid_samples,lower_rehearsal_grid_samples


class GatherGuards(unittest.TestCase):
    def test_both_entry_points_reject_unregistered_graph_without_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
            path=Path(root)/'unknown.onnx';original=b'not an admitted graph';path.write_bytes(original)
            for lower in [lower_student_grid_samples,lower_rehearsal_grid_samples]:
                with self.assertRaises(ValueError):lower(path)
                self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
