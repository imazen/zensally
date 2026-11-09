from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api_budget import configure,cost,guarded
from sunburst_generate import append


class Reply:
    headers={}
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,*args):return json.dumps(self.value).encode()


class ResponseBudget(unittest.TestCase):
    def authorization(self,root):
        path=root/'authorization.json';path.write_text(json.dumps(dict(schema='additional-crop-experiment-v4',additional_total_usd='20',
            generation_ceiling_usd='12',testing_ceiling_usd='8',owner_instruction='You have permission to spend $20 on more image generation and testing.')));return path

    def request(self):return dict(model='gpt-6.1-terra',input='fixture',max_output_tokens=100,text={})
    def task(self):return dict(id='fixture',role='fit',image=dict(sha256='fixture-sha'))

    def opener(self,calls):
        def call(request,**kwargs):
            calls.append(request.full_url)
            if request.full_url.endswith('/input_tokens'):return Reply(dict(object='response.input_tokens',input_tokens=500))
            return Reply(dict(id='fake-response',status='completed',model='gpt-6.1-terra',output=[],usage=dict(input_tokens=500,output_tokens=10,input_tokens_details={})))
        return call

    def test_native_cache_rates_are_not_double_counted(self):
        self.assertEqual(cost(dict(input_tokens=1000,output_tokens=500,input_tokens_details=dict(cached_tokens=100,cache_write_tokens=200)),'gpt-6.1-terra'),Decimal('.003455'))

    def test_replay_has_no_count_or_inference_request_or_extra_charge(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);authorization=self.authorization(root);calls=[]
            with patch.dict(os.environ,{'OPENAI_API_KEY':'fixture'}):
                result=guarded(self.task(),self.request(),lambda *_:{},root/'budget',root/'cache',authorization,opener=self.opener(calls))
            self.assertEqual(len(calls),2);self.assertEqual(result['new_charge_usd'],'0.00055')
            before=(root/'budget/budget.jsonl').read_bytes()
            replay=guarded(self.task(),self.request(),lambda *_:{},root/'budget',root/'cache',authorization,opener=lambda *_a,**_kw:self.fail('cache replay contacted provider'))
            self.assertEqual(replay['new_charge_usd'],'0');self.assertEqual((root/'budget/budget.jsonl').read_bytes(),before)

    def test_exhausted_budget_refuses_before_model_inference(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);authorization=self.authorization(root);configure(root/'budget',authorization)
            append(root/'budget/budget.jsonl',dict(key='prior',state='reserved',usd='7.999'))
            calls=[]
            with patch.dict(os.environ,{'OPENAI_API_KEY':'fixture'}),self.assertRaisesRegex(ValueError,'before inference'):
                guarded(self.task(),self.request(),lambda *_:{},root/'budget',root/'cache',authorization,opener=self.opener(calls))
            self.assertEqual(calls,['https://api.openai.com/v1/responses/input_tokens'])


if __name__=='__main__':unittest.main()
