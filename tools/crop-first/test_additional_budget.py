from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from sunburst_generate import configure_budget


class AdditionalBudget(unittest.TestCase):
    def authorization(self,root):
        data=dict(schema='additional-crop-experiment-v4',additional_total_usd='20',generation_ceiling_usd='12',testing_ceiling_usd='8',
            owner_instruction='You have permission to spend $20 on more image generation and testing.')
        path=root/'authorization.json';path.write_text(json.dumps(data));return path

    def test_higher_limit_requires_the_new_explicit_authorization(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):configure_budget(root/'budget.jsonl',Decimal('12'))
            path=self.authorization(root);configure_budget(root/'budget.jsonl',Decimal('12'),authorization=path)
            config=json.loads((root/'budget-config.json').read_text())
            self.assertEqual(config['ceiling_usd'],'12');self.assertIn('authorization_sha256',config)

    def test_split_cannot_exceed_the_authorized_total(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);path=self.authorization(root);data=json.loads(path.read_text());data['testing_ceiling_usd']='9';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):configure_budget(root/'budget.jsonl',Decimal('12'),authorization=path)
            self.assertFalse((root/'budget-config.json').exists())


if __name__=='__main__':unittest.main()
