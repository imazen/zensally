#!/usr/bin/env python3
"""Recorded Terra teacher tokens priced as an API comparison, never an invoice.

Rates verified 2026-11-07 at the linked official model page. Only standard,
short-context, zero-cache-write records are covered; other meters fail closed.
"""
import argparse,json
from decimal import Decimal
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollout',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows=[json.loads(s) for s in a.rollout.read_text().splitlines()]
    maximum=0;usage=None;previous={};turns=[]
    for r in rows:
        v=r.get('payload',{})
        if r['type']=='turn_context' and v.get('model')!='gpt-6.1-terra':raise ValueError('unexpected model in teacher session')
        if r['type']=='event_msg' and v.get('type')=='token_count':
            maximum=max(maximum,v['info']['last_token_usage']['input_tokens']);usage=v['info']['total_token_usage']
        if r['type']=='event_msg' and v.get('type')=='task_complete':
            if usage is None:raise ValueError('usage missing for completed turn')
            delta={k:usage[k]-previous.get(k,0) for k in usage};previous=usage.copy()
            if delta.get('cache_write_input_tokens',0)!=0 or maximum>272000:raise ValueError('additional rate treatment required')
            equivalent=(Decimal(delta['input_tokens']-delta['cached_input_tokens'])+Decimal(delta['cached_input_tokens'])*Decimal('.05')+Decimal(delta['output_tokens'])*5)/1000000
            turns.append({'completed_at':r['timestamp'],'usage_including_cli_context':delta,'standard_api_equivalent_usd':str(equivalent)})
    if not turns:raise ValueError('no completed teacher turns')
    result={'scope':'this named Terra teacher session only; excludes coordinator and other work','model':'gpt-6.1-terra',
        'transport':'signed-in Codex CLI','billed_amount_known':False,'direct_api_requests_from_adapter':0,
        'pricing_url':'https://developers.openai.com/api/docs/models/gpt-6.1-terra','rates_verified':'2026-11-07',
        'rates_usd_per_million':{'uncached_input':'1.00','cached_input':'.05','output':'5.00'},
        'max_single_request_input_tokens':maximum,'total_usage':previous,
        'standard_api_equivalent_usd':str(sum(Decimal(t['standard_api_equivalent_usd']) for t in turns)),
        'turns':turns,'warning':'API comparison only; no verified subscription credit charge, allowance balance or actual API invoice. No extrapolation to a larger run.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['standard_api_equivalent_usd','billed_amount_known','total_usage']}))


if __name__=='__main__':main()
