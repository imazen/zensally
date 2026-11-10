import unittest
from copy import deepcopy
from attention import explicit_heads


class SuppliedHeads(unittest.TestCase):
    def test_source_identity_bounds_and_explicit_empty_inputs(self):
        row=dict(id='a',role='fit',family_id='f',rgb_sha256='abc')
        entry={k:row[k] for k in ['role','family_id','rgb_sha256']};entry['people']=[dict(head=[.1,.2,.4,.5])]
        self.assertEqual(explicit_heads(row,{'a':entry}),entry['people'])
        for key,value in [('role','dev'),('family_id','other'),('rgb_sha256','different')]:
            changed=deepcopy(entry);changed[key]=value
            with self.assertRaises(ValueError):explicit_heads(row,{'a':changed})
        for box in [[0,0,0,1],[-.1,0,1,1],[0,0,float('nan'),1]]:
            changed=deepcopy(entry);changed['people'][0]['head']=box
            with self.assertRaises(ValueError):explicit_heads(row,{'a':changed})
        entry['people']=[];self.assertEqual(explicit_heads(row,{'a':entry}),[])
        entry['people']=[dict(head=[0,0,1,1])]*17
        with self.assertRaises(ValueError):explicit_heads(row,{'a':entry})


if __name__=='__main__':unittest.main()
