import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analyze_subject_sets import vectors


class InitializerBaseline(unittest.TestCase):
    def test_selected_improvement_cannot_replace_epoch_zero(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as scratch:
            root=Path(scratch)
            for seed in [31,137,911]:
                directory=root/f'trainable-image-seed{seed}';directory.mkdir()
                for filename,match,cut in [('predictions-000.json',False,True),('best-predictions.json',True,False)]:
                    rows=[dict(essential=True,family_id='a',covered=True,reference_match=match,explicit_face_cut=cut),
                        dict(essential=False,family_id='a',covered=False,reference_match=False,explicit_face_cut=False)]
                    (directory/filename).write_text(json.dumps(dict(new=rows)))
            families,initial=vectors(root,'trainable-image','predictions-000.json')
            selected_families,selected=vectors(root,'trainable-image')
            self.assertEqual(families,selected_families)
            np.testing.assert_array_equal(initial,[[1.,0.,1.]])
            np.testing.assert_array_equal(selected,[[1.,1.,0.]])


if __name__=='__main__':unittest.main()
