#!/usr/bin/env python3
"""Approved project Images API runner: immutable responses and budget admission."""
import argparse
import base64
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sunburst_costs import MODEL, MAX_HIGH_OUTPUT_TOKENS, dimensions, output_tokens

RESERVATION = Decimal('.25')
LIMIT = Decimal('.95')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload(job):
    dimensions(job['size'])
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,90}', job['id']):
        raise ValueError('invalid image identity')
    if not 1 <= len(job['prompt'].encode()) <= 8000:
        raise ValueError('prompt must contain 1..8000 UTF-8 bytes')
    return {'model': MODEL, 'quality': 'high', 'complexity': 'high', 'n': 1,
            'size': job['size'], 'prompt': job['prompt'], 'background': 'opaque',
            'output_format': 'png'}


def usage_cost(usage):
    details = usage['input_tokens_details']
    values = [details['text_tokens'], details['image_tokens'], usage['input_tokens'], usage['output_tokens']]
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('invalid native usage counts')
    if details['image_tokens'] or usage['input_tokens'] != details['text_tokens']:
        raise ValueError('unexpected image/context input in text-only generation')
    return (Decimal(details['text_tokens']) * 5 + Decimal(usage['output_tokens']) * 30) / 1_000_000


def journal(path):
    states = {}
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            states[event['key']] = event
    return states


def exposure(states):
    return sum((Decimal(v['usd']) for v in states.values()), Decimal(0))


def append(path, event):
    with path.open('ab') as f:
        f.write(canonical(event) + b'\n')
        f.flush()
        os.fsync(f.fileno())


def decode(entry, request):
    response_bytes = (entry / 'response.json').read_bytes()
    meta = json.loads((entry / 'completed.json').read_text())
    if meta['response_sha256'] != sha(response_bytes) or (entry / 'request.json').read_bytes() != canonical(request):
        raise ValueError('cached request/response changed')
    response = json.loads(response_bytes)
    if len(response['data']) != 1:
        raise ValueError('expected exactly one generated image')
    png = base64.b64decode(response['data'][0]['b64_json'], validate=True)
    if not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('expected native PNG output')
    return png, response, meta


def configure_budget(ledger, ceiling, carry=None):
    """Bind an approved ceiling and optional settled predecessor once, under lock."""
    if not ceiling.is_finite() or not Decimal('.25') < ceiling <= Decimal('2.50'):
        raise ValueError('invalid approved budget ceiling')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    config = {'ceiling_usd': str(ceiling)}
    carry_event = None
    if carry:
        raw = carry.read_bytes()
        states = journal(carry)
        if not states or any(v['state'] != 'settled' for v in states.values()):
            raise ValueError('carry journal must contain only settled attempts')
        config.update(carry_sha256=sha(raw), carry_usd=str(exposure(states)))
        carry_event = {'key': 'carry-' + sha(raw), 'state': 'settled',
                       'usd': config['carry_usd'], 'journal_sha256': sha(raw)}
    with (ledger.parent / (ledger.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = ledger.parent / 'budget-config.json'
        if target.exists():
            if json.loads(target.read_text()) != config:
                raise ValueError('budget configuration changed')
        else:
            if ledger.exists():
                raise ValueError('cannot reconfigure an existing unbound ledger')
            target.write_bytes(canonical(config))
        if carry_event:
            states = journal(ledger)
            if carry_event['key'] not in states:
                if states:
                    raise ValueError('carry must precede new paid attempts')
                append(ledger, carry_event)


def generate(job, cache, ledger, *, opener=urlopen, token=None, ceiling=LIMIT):
    request = payload(job)
    key = sha(canonical(request))
    cache.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = cache / key
    # The ledger lock serializes this experiment's monetary reservations only.
    with (ledger.parent / (ledger.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states = journal(ledger)
        if (entry / 'completed.json').exists():
            png, response, meta = decode(entry, request)
            return png, response, {**meta, 'cache_hit': True, 'key': key, 'exposure_usd': str(exposure(states))}
        if key in states or entry.exists():
            raise ValueError('unresolved attempt retained; refusing an automatic repeated POST')
        if exposure(states) + RESERVATION >= ceiling:
            raise ValueError('budget admission refused before network request')
        token = token or os.environ.get('OPENAI_API_KEY')
        if not token:
            raise ValueError('dev API key required for an uncached generation')
        entry.mkdir()
        (entry / 'request.json').write_bytes(canonical(request))
        append(ledger, {'key': key, 'state': 'reserved', 'usd': str(RESERVATION), 'created_ns': time.time_ns()})
        req = Request('https://api.openai.com/v1/images/generations', data=canonical(request),
                      headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        try:
            with opener(req, timeout=600) as reply:
                raw = reply.read(64 * 1024 * 1024 + 1)
                request_id = reply.headers.get('x-request-id')
        except HTTPError as error:
            (entry / 'http-error.json').write_bytes(error.read())
            raise
        (entry / 'response.json').write_bytes(raw)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError('response exceeds bounded capture; reservation retained')
        response = json.loads(raw)
        actual = usage_cost(response['usage'])
        meta = {'model': MODEL, 'quality': 'high', 'complexity': 'high',
                'request_id': request_id, 'response_sha256': sha(raw),
                'published_rate_usd': str(actual), 'invoice_verified': False,
                'calculator_output_tokens': output_tokens(job['size']),
                'native_output_tokens': response['usage']['output_tokens'],
                'synthetic': True, 'training_eligible': False}
        (entry / 'completed.json').write_bytes(canonical(meta))
        append(ledger, {'key': key, 'state': 'settled', 'usd': str(actual), 'created_ns': time.time_ns()})
        if actual > RESERVATION or response['usage']['output_tokens'] > MAX_HIGH_OUTPUT_TOKENS:
            raise ValueError('native charge exceeded published-envelope reservation; stop and inspect')
        png, response, meta = decode(entry, request)
        return png, response, {**meta, 'cache_hit': False, 'key': key,
                               'exposure_usd': str(exposure(journal(ledger)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('cache', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--budget-ceiling-usd', type=Decimal, default=LIMIT)
    p.add_argument('--carry-journal', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(a.plan.read_text())
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('plan image identities must be unique')
    for job in jobs:
        payload(job)
    if a.carry_journal or a.budget_ceiling_usd != LIMIT:
        configure_budget(a.output / 'budget.jsonl', a.budget_ceiling_usd, a.carry_journal)
    for job in jobs[:a.limit] if a.limit else jobs:
        png, response, meta = generate(job, a.cache, a.output / 'budget.jsonl', ceiling=a.budget_ceiling_usd)
        target = a.output / (job['id'] + '.png')
        if target.exists() and sha(target.read_bytes()) != sha(png):
            raise ValueError('existing output differs from immutable generated source')
        if not target.exists():
            target.write_bytes(png)
        record = {**job, **meta, 'png_sha256': sha(png), 'usage': response['usage']}
        evidence = a.output / (job['id'] + '.json')
        if not evidence.exists():
            evidence.write_bytes(canonical(record))
        print(json.dumps({'id': job['id'], 'size': job['size'], **meta}), flush=True)


if __name__ == '__main__':
    main()
