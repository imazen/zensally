import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from direct_zones import call_cached,RATIOS

class DirectZonesTests(unittest.TestCase):
    def run_fixture(self,root):
        image=root/'source.png';image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
        task={'id':'fixture','role':'train','width':1200,'height':800,'image':{'path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}}
        zones={'source_id':'fixture','face_coverage':'none_visible','rois':[],'crops':[{'ratio':r,'mode':m,'action':'contain','options':[],'reason':'fixture'} for r in RATIOS for m in ['minimal','maximal']]}
        response={'id':'resp_fixture','model':'gpt-6.1-terra','status':'completed','usage':{'input_tokens':100,'output_tokens':20},'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(zones)}]}]}
        return task,response

    def scratch(self):
        parent=Path.home()/'tmp';parent.mkdir(exist_ok=True);return tempfile.TemporaryDirectory(dir=parent)

    def opener(self,response,calls):
        class Reply:
            headers={'x-request-id':'request_fixture'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(response).encode()
        def open_request(req,timeout):
            calls.append(req.data)
            self.assertEqual(req.full_url,'https://api.openai.com/v1/responses')
            self.assertEqual(json.loads(req.data)['tool_choice'],'none')
            return Reply()
        return open_request

    def test_identical_request_reuses_response_without_key_or_network(self):
        with self.scratch() as name,patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-placeholder'}):
            root=Path(name);task,response=self.run_fixture(root);calls=[]
            first=call_cached(task,root/'cache',opener=self.opener(response,calls))
            del os.environ['OPENAI_API_KEY']
            second=call_cached(task,root/'cache',opener=lambda *_:self.fail('cache hit called network'))
            self.assertEqual(len(calls),1);self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit']);self.assertEqual(second['network_requests'],0)
            self.assertEqual(first['cache_key'],second['cache_key'])
            self.assertNotIn('test-only-placeholder',''.join(p.read_text() for p in (root/'cache').rglob('*.json')))

    def test_changed_image_changes_identity_and_tampering_fails_closed(self):
        with self.scratch() as name,patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-placeholder'}):
            root=Path(name);task,response=self.run_fixture(root);calls=[];opener=self.opener(response,calls)
            first=call_cached(task,root/'cache',opener=opener)
            Path(task['image']['path']).write_bytes(b'\x89PNG\r\n\x1a\nchanged')
            task['image']['sha256']=hashlib.sha256(Path(task['image']['path']).read_bytes()).hexdigest()
            second=call_cached(task,root/'cache',opener=opener)
            self.assertNotEqual(first['cache_key'],second['cache_key']);self.assertEqual(len(calls),2)
            (Path(second['entry'])/'response.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'cached response bytes changed'):call_cached(task,root/'cache',opener=opener)
            self.assertEqual(len(calls),2)

    def test_incomplete_response_is_retained_but_not_cached(self):
        with self.scratch() as name,patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-placeholder'}):
            root=Path(name);task,response=self.run_fixture(root);response['status']='incomplete';calls=[]
            with self.assertRaisesRegex(ValueError,'incomplete'):call_cached(task,root/'cache',opener=self.opener(response,calls))
            self.assertEqual(len(calls),1);self.assertEqual(len(list((root/'cache/attempts').glob('*/response.json'))),1)
            self.assertEqual([p for p in (root/'cache').iterdir() if p.is_dir() and p.name!='attempts'],[])

if __name__=='__main__':unittest.main()
