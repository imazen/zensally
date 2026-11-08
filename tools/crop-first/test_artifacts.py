import hashlib
import tempfile
import unittest
from pathlib import Path

from artifacts import verify


class ArtifactTests(unittest.TestCase):
    def test_both_manifests_reject_content_and_membership_changes(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            file = root/'input'
            file.write_bytes(b'abc')
            digest = hashlib.sha256(b'abc').hexdigest()
            for expected in [{'input': digest}, [{'path': 'input', 'bytes': 3, 'sha256': digest}]]:
                self.assertEqual(verify(root, expected), 1)
                file.write_bytes(b'abd')
                with self.assertRaises(ValueError): verify(root, expected)
                file.write_bytes(b'abc')
                (root/'extra').write_text('x')
                with self.assertRaises(ValueError): verify(root, expected)
                (root/'extra').unlink()

    def test_rich_manifest_rejects_size_duplicate_and_unknown_fields(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            (root/'input').write_bytes(b'abc')
            row = {'path': 'input', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
            for expected in [[dict(row, bytes=4)], [row, row], [dict(row, extra=True)]]:
                with self.assertRaises(ValueError): verify(root, expected)


if __name__ == '__main__':
    unittest.main()
