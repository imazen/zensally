import base64
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from sunburst_costs import dimensions, output_tokens, table
from sunburst_generate import append, canonical, configure_budget, exposure, generate, journal, sha, usage_cost


class SunburstTests(unittest.TestCase):
    def test_calculator_comparisons_and_orientation_symmetry(self):
        self.assertEqual(output_tokens('1024x1024'), 1756)
        self.assertEqual(output_tokens('2048x1024'), 1180)
        self.assertEqual(output_tokens('3072x1024'), 988)
        self.assertEqual(output_tokens('1024x3072'), 988)
        self.assertEqual(output_tokens('2880x2880'), 5930)
        self.assertEqual(output_tokens('944x1536'), 1201)
        self.assertEqual(output_tokens('1888x3072'), 2715)
        minima = {r['requested_ratio']: r['lowest_cost_exact_ratio'] for r in table()['exact_ratios']}
        self.assertEqual(minima['3:1']['size'], '1440x480')
        self.assertEqual(minima['4:3']['size'], '960x720')
        self.assertEqual(minima['1:3']['calculator_output_tokens'], 517)
        for size in ['1024x512', '1025x1024', '1024x4096', '0x1024']:
            with self.assertRaises(ValueError):
                dimensions(size)

    def test_revised_commerce_scope_and_paid_request_envelope(self):
        from sunburst_commerce_v2 import plan
        from sunburst_generate import payload
        jobs = plan()
        commerce, candids = jobs[:20], jobs[20:]
        self.assertEqual(len(jobs), 32)
        self.assertEqual(sum(j['background_requested'] == 'white' for j in commerce), 16)
        self.assertEqual(sum(j['size'] in ['1536x1152', '1152x1536'] for j in commerce), 10)
        self.assertEqual(sum(j['size'] == '1152x1536' for j in candids), 8)
        self.assertEqual(len(candids), 12)
        self.assertTrue(all(j['brand_requested'] for j in commerce))
        self.assertTrue(all(not j['training_eligible'] for j in jobs))
        for job in jobs:
            payload(job)

    def test_gallery_honors_bound_total_including_previous_generation(self):
        import contextlib, io, sys
        from unittest.mock import patch
        import sunburst_gallery
        with self.scratch() as name:
            root = Path(name)
            job = {'id': 'fixture', 'size': '1536x1152'}
            (root / 'plan.json').write_text(json.dumps([job]))
            (root / 'fixture.png').write_bytes(b'fixture')
            record = {**job, 'png_sha256': sha(b'fixture'), 'published_rate_usd': '1.45',
                      'product': 'fixture', 'crop_challenge': 'fixture', 'people_requested': 0,
                      'animals_requested': 0, 'scenario_family': 'fixture', 'prompt': 'photorealistic'}
            (root / 'fixture.json').write_text(json.dumps(record))
            (root / 'review.json').write_text(json.dumps({'notes': {'fixture': 'fixture'}}))
            config = root / 'budget-config.json'
            config.write_text(json.dumps({'ceiling_usd': '2.45', 'carry_usd': '.637020'}))
            with patch.object(sys, 'argv', ['gallery', str(root)]), contextlib.redirect_stdout(io.StringIO()):
                sunburst_gallery.main()
            self.assertIn('$2.087020 including prior usage', (root / 'index.html').read_text())
            config.write_text(json.dumps({'ceiling_usd': '2.45', 'carry_usd': '1.00'}))
            with patch.object(sys, 'argv', ['gallery', str(root)]), self.assertRaisesRegex(ValueError, 'budget'):
                sunburst_gallery.main()

    def test_sized_reservations_preserve_default_and_bind_policy(self):
        from sunburst_generate import RESERVATION, reservation_for
        from sunburst_products_selfies_v3 import plan
        for job in plan():
            self.assertEqual(reservation_for(job), RESERVATION)
            self.assertLess(reservation_for(job, True), RESERVATION)
            self.assertGreater(reservation_for(job, True), Decimal(output_tokens(job['size'])) * 30 / 1_000_000)
        with self.scratch() as name:
            root = Path(name)
            job, response = self.fixture()
            calls = []
            generate(job, root / 'cache', root / 'budget.jsonl', opener=self.opener(response, calls), token='fixture', sized=True)
            events = [json.loads(line) for line in (root / 'budget.jsonl').read_text().splitlines()]
            self.assertEqual(Decimal(events[0]['usd']), reservation_for(job, True))
            self.assertEqual(events[-1]['state'], 'settled')
            ledger = root / 'other' / 'budget.jsonl'
            configure_budget(ledger, Decimal('2.45'), sized=True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                configure_budget(ledger, Decimal('2.45'))

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

    def test_approved_budget_carries_prior_spend_once_and_binds_configuration(self):
        with self.scratch() as name:
            root = Path(name)
            old = root / 'old.jsonl'
            append(old, {'key': 'old', 'state': 'settled', 'usd': '.637020'})
            ledger = root / 'new' / 'budget.jsonl'
            configure_budget(ledger, Decimal('2.45'), old)
            configure_budget(ledger, Decimal('2.45'), old)
            self.assertEqual(exposure(journal(ledger)), Decimal('.637020'))
            with self.assertRaisesRegex(ValueError, 'changed'):
                configure_budget(ledger, Decimal('2.50'), old)
            append(old, {'key': 'unknown', 'state': 'reserved', 'usd': '.25'})
            with self.assertRaisesRegex(ValueError, 'settled'):
                configure_budget(ledger, Decimal('2.45'), old)

    def test_invalid_usage_is_not_silently_treated_as_zero(self):
        _, response = self.fixture()
        response['usage']['input_tokens_details']['image_tokens'] = 1
        with self.assertRaises(ValueError):
            usage_cost(response['usage'])


if __name__ == '__main__':
    unittest.main()
