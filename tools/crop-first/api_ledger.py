#!/usr/bin/env python3
"""Count unique completed responses, including invalid annotations; not invoices."""
import argparse,json,re
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
from corpus import sha

RATES={'gpt-6.1-terra':{'input':'1','read':'.05','write':'1.25','output':'5'},'gpt-6.1-sol':{'input':'2','read':'.10','write':'2.50','output':'10'}}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args();totals=defaultdict(Counter);seen=set()
    for path in a.cache.glob('*/metadata.json'):
        if not re.fullmatch('[0-9a-f]{64}',path.parent.name):continue
        entry=path.parent
        for name,digest in json.loads((entry/'hashes.json').read_text()).items():
            if sha((entry/name).read_bytes())!=digest:raise ValueError('cache payload changed')
        metadata=json.loads(path.read_text());response=json.loads((entry/'response.json').read_text());rid=response['id']
        if rid in seen:continue
        seen.add(rid);t=totals[response['model']];u=response['usage'];d=u.get('input_tokens_details',{})
        if u['input_tokens']>272_000:raise ValueError('this ledger requires the separately documented long-context rates')
        t['responses']+=1;t['input_tokens']+=u['input_tokens'];t['cache_read_tokens']+=d.get('cached_tokens',0);t['cache_write_tokens']+=d.get('cache_write_tokens',0);t['output_tokens']+=u['output_tokens'];t['invalid_annotations']+=not metadata['annotation_valid']
    results={};total=Decimal(0)
    for model,t in totals.items():
        t['uncached_input_tokens']=t['input_tokens']-t['cache_read_tokens']-t['cache_write_tokens']
        if t['uncached_input_tokens']<0:raise ValueError('cache accounting overlaps input')
        rates=RATES[model];cost=sum(Decimal(t[k])*Decimal(rates[r]) for k,r in [('uncached_input_tokens','input'),('cache_read_tokens','read'),('cache_write_tokens','write'),('output_tokens','output')])/Decimal(1_000_000);total+=cost;results[model]={**dict(t),'published_rate_usd':str(cost),'rates_per_million':rates}
    result={'models':results,'unique_completed_responses':len(seen),'published_rate_total_usd':str(total),'invoice_verified':False,'rates_checked':'2026-11-08','context_under_272k':True,'standard_cache_write_retention':'30 minutes; request does not override retention','links':[{'text':'Terra model and pricing','href':'https://developers.openai.com/api/docs/models/gpt-6.1-terra'},{'text':'Sol model and pricing','href':'https://developers.openai.com/api/docs/models/gpt-6.1-sol'},{'text':'Prompt caching','href':'https://developers.openai.com/api/docs/guides/prompt-caching'}]}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
