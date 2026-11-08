#!/usr/bin/env python3
"""Recorded Terra teacher tokens priced as an API comparison, never an invoice.

Rates verified 2026-11-08 at the linked official model page. Standard API comparison with per-request long-context adjustments.
Cache-write pricing requires an explicit retention policy and fails closed.
"""
import argparse,json
from decimal import Decimal
from pathlib import Path
from cli_records import read_records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollout',type=Path);p.add_argument('output',type=Path);p.add_argument('--model',choices=['gpt-6.1-terra','gpt-6-astra'],default='gpt-6.1-terra');a=p.parse_args()
    rows=read_records(a.rollout)
    maximum=0;usage=None;previous={};request_previous={};turns=[];turn_cost=Decimal(0);requests=0;long_requests=0
    rates=(Decimal('1'),Decimal('.05'),Decimal('5')) if a.model=='gpt-6.1-terra' else (Decimal('10'),Decimal('1'),Decimal('50'))
    for r in rows:
        v=r.get('payload',{})
        if r['type']=='turn_context' and v.get('model')!=a.model:raise ValueError('unexpected model in teacher session')
        if r['type']=='event_msg' and v.get('type')=='token_count':
            usage=v['info']['total_token_usage']
            delta={k:usage[k]-request_previous.get(k,0) for k in usage}
            if any(v<0 for v in delta.values()):raise ValueError('usage counters reset; manual reconciliation required')
            if any(delta.values()):
                if delta['input_tokens']!=v['info']['last_token_usage']['input_tokens']:raise ValueError('non-atomic request usage; manual reconciliation required')
                maximum=max(maximum,delta['input_tokens'])
                if delta.get('cache_write_input_tokens',0)!=0:raise ValueError('cache-write retention pricing is not established')
                long=delta['input_tokens']>272000;requests+=1;long_requests+=int(long)
                turn_cost+=((Decimal(delta['input_tokens']-delta['cached_input_tokens'])*rates[0]+Decimal(delta['cached_input_tokens'])*rates[1])*(2 if long else 1)+Decimal(delta['output_tokens'])*rates[2]*(Decimal('1.5') if long else 1))/1000000
                request_previous=usage.copy()
        if r['type']=='event_msg' and v.get('type')=='task_complete':
            if usage is None:raise ValueError('usage missing for completed turn')
            delta={k:usage[k]-previous.get(k,0) for k in usage};previous=usage.copy()
            equivalent=turn_cost;turn_cost=Decimal(0)
            turns.append({'completed_at':r['timestamp'],'usage_including_cli_context':delta,'standard_api_equivalent_usd':str(equivalent)})
    if not turns:raise ValueError('no completed teacher turns')
    result={'scope':'this named '+a.model+' CLI session only; excludes coordinator and other work','model':a.model,
        'transport':'signed-in Codex CLI','billed_amount_known':False,'api_invoice_available':False,
        'pricing_url':'https://developers.openai.com/api/docs/models/'+a.model,'rates_verified':'2026-11-08',
        'rates_usd_per_million':dict(zip(['uncached_input','cached_input','output'],map(str,rates))),
        'requests':requests,'long_context_requests':long_requests,'max_single_request_input_tokens':maximum,'total_usage':previous,
        'standard_api_equivalent_usd':str(sum(Decimal(t['standard_api_equivalent_usd']) for t in turns)),
        'turns':turns,'warning':'API comparison only; no verified subscription credit charge, allowance balance or actual API invoice. No extrapolation to a larger run.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['standard_api_equivalent_usd','billed_amount_known','total_usage']}))


if __name__=='__main__':main()
