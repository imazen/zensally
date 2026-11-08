import base64
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from sunburst_costs import dimensions, output_tokens, table
from sunburst_generate import append, canonical, generate, sha, usage_cost


class SunburstTests(unittest.TestCase):
    def test_calculator_comparisons_and_orientation_symmetry(self):
        self.assertEqual(output_tokens('1024x1024'), 1756)
        self.assertEqual(output_tokens('2048x1024'), 1180)
        self.assertEqual(output_tokens('3072x1024'), 988)
        self.assertEqual(output_tokens('1024x3072'), 988)
        self.assertEqual(output_tokens('2880x2880'), 5930)
        minima = {r['requested_ratio']: r['lowest_cost_exact_ratio'] for r in table()['exact_ratios']}
        self.assertEqual(minima['3:1']['size'], '1440x480')
        self.assertEqual(minima['4:3']['size'], '960x720')
        self.assertEqual(minima['1:3']['calculator_output_tokens'], 517)
        for size in ['1024x512', '1025x1024', '1024x4096', '0x1024']:
            with self.assertRaises(ValueError):
                dimensions(size)

    def fixture(self):
        job = {'id': 'fixture', 'size': '3072x1024', 'prompt': 'A product photograph.'}
        response = {'data': [{'b64_json': base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()}],
                    'usage': {'input_tokens': 10, 'input_tokens_details': {'text_tokens': 10, 'image_tokens': 0}, 'output_tokens': 988}}
        return job, response

    def scratch(self):
        parent = Path.home() / 'tmp'
        parent.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def opener(self, response, calls):
        class Reply:
            headers = {'x-request-id': 'fixture-request'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return canonical(response)
        def call(request, timeout):
            calls.append(json.loads(request.data))
            return Reply()
        return call

    def test_cached_response_and_usage_are_retained_without_another_charge(self):
        with self.scratch() as name:
            root = Path(name)
            job, response = self.fixture()
            calls = []
            first = generate(job, root / 'cache', root / 'budget.jsonl', opener=self.opener(response, calls), token='fixture-key')
            second = generate(job, root / 'cache', root / 'budget.jsonl', opener=lambda *_: self.fail('repeated POST'))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['model'], 'gpt-image-2.6-sunburst')
            self.assertEqual(calls[0]['quality'], 'high')
            self.assertEqual(first[0], second[0])
            self.assertTrue(second[2]['cache_hit'])
            self.assertEqual(Decimal(first[2]['exposure_usd']), Decimal('.02969'))
            raw = next((root / 'cache').glob('*/response.json'))
            self.assertEqual(json.loads(raw.read_text()), response)
            self.assertNotIn('fixture-key', ''.join(p.read_text() for p in root.rglob('*.json')))
            raw.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                generate(job, root / 'cache', root / 'budget.jsonl')

    def test_budget_refuses_before_network_and_failed_attempt_is_never_reposted(self):
        with self.scratch() as name:
            root = Path(name)
            job, _ = self.fixture()
            ledger = root / 'budget.jsonl'
            append(ledger, {'key': 'prior', 'state': 'settled', 'usd': '.71'})
            with self.assertRaisesRegex(ValueError, 'budget admission'):
                generate(job, root / 'cache', ledger, opener=lambda *_: self.fail('over-budget POST'), token='fixture')
        with self.scratch() as name:
            root = Path(name)
            def timeout(*args, **kwargs): raise TimeoutError('unknown provider completion')
            with self.assertRaises(TimeoutError):
                generate(job, root / 'cache', root / 'budget.jsonl', opener=timeout, token='fixture')
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                generate(job, root / 'cache', root / 'budget.jsonl', opener=lambda *_: self.fail('automatic repeated POST'))

    def test_invalid_usage_is_not_silently_treated_as_zero(self):
        _, response = self.fixture()
        response['usage']['input_tokens_details']['image_tokens'] = 1
        with self.assertRaises(ValueError):
            usage_cost(response['usage'])


if __name__ == '__main__':
    unittest.main()
