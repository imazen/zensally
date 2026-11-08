import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from direct_zones import call_cached,RATIOS,build

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

    def test_custom_ratios_and_independent_review_model_have_distinct_cached_requests(self):
        with self.scratch() as name,patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-placeholder'}):
            root=Path(name);task,response=self.run_fixture(root);task['ratios']=['1:3','3:1','1:6','6:1'];calls=[]
            zones=json.loads(response['output'][0]['content'][0]['text']);zones['crops']=[{'ratio':r,'mode':m,'action':'contain','options':[],'reason':'fixture'} for r in task['ratios'] for m in ['minimal','maximal']]
            response['output'][0]['content'][0]['text']=json.dumps(zones)
            terra=call_cached(task,root/'cache',opener=self.opener(response,calls))
            request=build(task);request['model']='gpt-6.1-sol'
            sol=call_cached(task,root/'cache',request=request,opener=self.opener(response,calls))
            hit=call_cached(task,root/'cache',request=request,opener=lambda *_:self.fail('review cache repeated POST'))
            self.assertTrue(terra['annotation_valid']);self.assertTrue(sol['annotation_valid']);self.assertNotEqual(terra['cache_key'],sol['cache_key']);self.assertTrue(hit['cache_hit']);self.assertEqual(len(calls),2)

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

    def test_full_source_contain_zone_is_valid_and_bad_completed_labels_still_cache(self):
        with self.scratch() as name,patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-placeholder'}):
            root=Path(name);task,response=self.run_fixture(root);calls=[]
            content=response['output'][0]['content'][0];zones=json.loads(content['text'])
            zones['crops'][0]['options']=[{'box':{'left':0,'top':0,'right':100,'bottom':100},'retained_roi_ids':[],'reason':'pad full source'}]
            content['text']=json.dumps(zones);one=call_cached(task,root/'valid',opener=self.opener(response,calls));self.assertTrue(one['annotation_valid'])
            zones['crops'][0]['options'][0]['box']['left']=10;content['text']=json.dumps(zones)
            bad=call_cached(task,root/'invalid',opener=self.opener(response,calls));self.assertFalse(bad['annotation_valid'])
            cached=call_cached(task,root/'invalid',opener=lambda *_:self.fail('repeated completed request'))
            self.assertFalse(cached['annotation_valid']);self.assertTrue(cached['cache_hit']);self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
