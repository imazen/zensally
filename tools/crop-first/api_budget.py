"""Additional Responses budget: token preflight, durable reservation, no reposts."""
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from api_ledger import RATES
from direct_zones import call_cached,canonical,sha
from sunburst_generate import append,budget_approval,exposure,journal

ASTRA_RATES={'input':'10','read':'1','write':'12.50','output':'50'}


def registered_rates(root,*,extend_astra=False):
    """Append a dated rate binding while preserving the original budget config."""
    path=root/'rates-extension-v1.json'
    expected=dict(parent_config_sha256=sha((root/'budget-config.json').read_bytes()),checked='2026-11-09',
        source='https://developers.openai.com/api/docs/models/gpt-6-astra',rates={'gpt-6-astra':ASTRA_RATES})
    with (root/'budget.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=expected:raise ValueError('rate extension binding differs')
        elif extend_astra:
            with path.open('x') as stream:stream.write(json.dumps(expected,indent=2)+'\n')
        else:return RATES
    return {**RATES,**expected['rates']}

def configure(root,authorization):
    if authorization is None:raise ValueError('additional testing authorization required')
    approval=json.loads(authorization.read_text())
    if approval.get('schema')=='crop-label-quality-authorization-v1':
        if (approval.get('owner_instruction')!='Is this a labeling quality issue? You can spend another $20 as long as you use it efficiently' or
                approval.get('additional_total_usd')!='20' or approval.get('stage_ceiling_usd')!='2'):
            raise ValueError('explicit20-dollar authorization and2-dollar label pilot required')
        binding=dict(authorization_sha256=sha(authorization.read_bytes()),additional_total_usd='20',scope='label-quality-pilot')
        ceiling='2'
    else:
        _,binding=budget_approval(authorization);ceiling='8'
    if not binding:raise ValueError('additional testing authorization required')
    expected=dict(binding,ceiling_usd=ceiling,rates_checked='2026-11-09',rates=RATES,retention='default30m')
    root.mkdir(parents=True,exist_ok=True)
    path=root/'budget-config.json'
    with (root/'budget.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=expected:raise ValueError('testing budget configuration differs')
        else:
            with path.open('x') as stream:stream.write(json.dumps(expected,indent=2)+'\n')
    return expected


def cost(usage,model,rates_map=None):
    rates=(RATES if rates_map is None else rates_map)[model];details=usage.get('input_tokens_details',{})
    n=usage['input_tokens'];read=details.get('cached_tokens',0);write=details.get('cache_write_tokens',0);out=usage['output_tokens']
    if any(type(v) is not int or v<0 for v in [n,read,write,out]) or read+write>n:raise ValueError('invalid native Responses token accounting')
    return sum(Decimal(tokens)*Decimal(rates[k]) for tokens,k in [(n-read-write,'input'),(read,'read'),(write,'write'),(out,'output')])/1_000_000


def preflight(request,root,key,*,opener=urlopen):
    directory=root/'token-counts';directory.mkdir(exist_ok=True);path=directory/(key+'.json')
    count_request={k:v for k,v in request.items() if k in ['model','input','instructions','tools','tool_choice','text','reasoning']}
    identity=sha(canonical(count_request))
    if path.exists():
        result=json.loads(path.read_text())
        if result['request_sha256']!=identity:raise ValueError('cached token-count contract differs')
        return result
    req=Request('https://api.openai.com/v1/responses/input_tokens',data=canonical(count_request),headers={
        'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
    try:
        with opener(req,timeout=60) as reply:raw=reply.read(1024*1024)
    except HTTPError as error:
        (directory/(key+'.http-error.json')).write_bytes(error.read());raise
    native=json.loads(raw);tokens=native['input_tokens']
    if type(tokens) is not int or not 0<tokens<272000:raise ValueError('bounded short-context token count required')
    result=dict(request_sha256=identity,response=native,input_tokens=tokens)
    with path.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


def guarded(task,request,validator,root,cache,authorization,*,opener=urlopen,extend_astra=False):
    config=configure(root,authorization);model=request['model']
    rates_map=registered_rates(root,extend_astra=extend_astra)
    if model not in rates_map or request.get('service_tier','default') not in ['default','standard']:
        raise ValueError('registered standard model pricing required')
    key=sha(canonical(dict(cache_protocol=1,request=request)));ledger=root/'budget.jsonl'
    with (root/'budget.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);states=journal(ledger)
        cached=(cache/key/'hashes.json').exists()
        if cached:
            result=call_cached(task,cache,request=request,validator=validator,opener=lambda *_args,**_kwargs:(_ for _ in ()).throw(ValueError('cache replay attempted network')))
            if key in states and states[key]['state']=='reserved':
                actual=cost(result['usage'],model,rates_map);append(ledger,dict(key=key,state='settled',usd=str(actual),model=model))
            return dict(result,new_charge_usd='0',budget_exposure_usd=str(exposure(journal(ledger))))
        if key in states:raise ValueError('unresolved paid attempt retained; automatic repost refused')
    count=preflight(request,root,key,opener=opener);rates=rates_map[model]
    # Count the complete schema and leave an additional framing-token envelope;
    # native usage is checked against this bound after the response is retained.
    input_bound=count['input_tokens']+len(canonical(request.get('text',{})))+2048
    output_bound=request['max_output_tokens']
    reservation=(Decimal(input_bound)*Decimal(rates['write'])+Decimal(output_bound)*Decimal(rates['output']))/1_000_000+Decimal('.005')
    with (root/'budget.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);states=journal(ledger)
        if key in states:raise ValueError('concurrent or unresolved request cannot be posted twice')
        if exposure(states)+reservation>=Decimal(config['ceiling_usd']):raise ValueError('testing budget refused before inference request')
        append(ledger,dict(key=key,state='reserved',usd=str(reservation),model=model,input_bound=input_bound,output_bound=output_bound))
    result=call_cached(task,cache,request=request,validator=validator,opener=opener)
    actual=cost(result['usage'],model,rates_map)
    with (root/'budget.jsonl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);append(ledger,dict(key=key,state='settled',usd=str(actual),model=model,annotation_valid=result['annotation_valid']))
    if actual>reservation or result['usage']['input_tokens']>input_bound or result['usage']['output_tokens']>output_bound:
        raise ValueError('native usage exceeded its retained budget envelope; stop and inspect')
    return dict(result,new_charge_usd=str(actual),budget_exposure_usd=str(exposure(journal(ledger))),preflight_input_tokens=count['input_tokens'])
