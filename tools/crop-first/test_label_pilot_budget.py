import json
from pathlib import Path
import tempfile
import unittest
from api_budget import configure


class LabelPilotBudget(unittest.TestCase):
    def test_new_authorization_is_bound_to_separate_stricter_pilot(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);p=root/'authorization.json'
            data=dict(schema='crop-label-quality-authorization-v1',additional_total_usd='20',stage_ceiling_usd='2',owner_instruction='Is this a labeling quality issue? You can spend another $20 as long as you use it efficiently')
            p.write_text(json.dumps(data));result=configure(root/'pilot',p)
            self.assertEqual(result['ceiling_usd'],'2');self.assertEqual(result['additional_total_usd'],'20')
            self.assertEqual(configure(root/'pilot',p),result)
            data['stage_ceiling_usd']='20';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):configure(root/'other',p)
            self.assertFalse((root/'other').exists())


if __name__=='__main__':unittest.main()
