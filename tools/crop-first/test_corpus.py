import json,unittest
from corpus import reconcile,select

class SplitTests(unittest.TestCase):
    def fixture(self):
        families=[{'id':'1000','family':'1000:burst','split':'train'},{'id':'1001','family':'1000:burst','split':'train'},{'id':'1003','family':'','split':'validate'}]
        def r(i,role):
            prov={'id':'1'+i,'split':role,'original_sha256':i,'category':'photo'}
            return {'id':'1'+i,'bucket':role,'original_sha256':i,'provenance_json':json.dumps(prov),'width':'640','height':'480'}
        return {'train':[r('1000','train'),r('1001','train')],'validate':[r('1003','validate')]},families
    def test_historical_eval_blocks_entire_new_train_family(self):
        e,f=self.fixture();r=reconcile(e,f,[{'id':'1001','split':'validate'}])
        self.assertTrue(all(x['historical_eval_family'] for x in r if x['bucket']=='train'))
        with self.assertRaises(ValueError):select(r,{'fit':1})
    def test_family_cannot_cross_fit_and_internal_dev(self):
        e,f=self.fixture();r=reconcile(e,f,[])
        with self.assertRaises(ValueError):select(r,{'fit':1,'dev':1})
    def test_export_role_conflict_rejected(self):
        e,f=self.fixture();f[0]['split']='test'
        with self.assertRaises(ValueError):reconcile(e,f,[])
    def test_current_validate_never_fits(self):
        e,f=self.fixture();r=reconcile(e,f,[]);s=select(r,{'fit':1,'eval':1})
        self.assertEqual([x['bucket'] for x in s],['train','validate'])
if __name__=='__main__':unittest.main()
