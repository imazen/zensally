#!/usr/bin/env python3
"""Cache-independent native usage verification for the funded Responses ledger."""
import argparse
from collections import Counter,defaultdict
from decimal import Decimal
import json
from pathlib import Path
from api_budget import cost,registered_rates
from sunburst_generate import journal,sha,exposure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','budget','output']:p.add_argument(key,type=Path)
    a=p.parse_args();rates=registered_rates(a.budget);states=journal(a.budget/'budget.jsonl');models=defaultdict(Counter);records=[];seen=set()
    for key,state in states.items():
        if state['state']!='settled':raise ValueError('unresolved monetary reservation retained; full settlement verification cannot pass')
        entry=a.cache/key
        for name,digest in json.loads((entry/'hashes.json').read_text()).items():
            if sha((entry/name).read_bytes())!=digest:raise ValueError('native cache hash differs')
        response=json.loads((entry/'response.json').read_text());metadata=json.loads((entry/'metadata.json').read_text())
        if response['id'] in seen:raise ValueError('provider response was charged twice in ledger')
        seen.add(response['id']);actual=cost(response['usage'],state['model'],rates)
        if actual!=Decimal(state['usd']):raise ValueError('native usage and settled charge differ')
        models[state['model']]['responses']+=1;models[state['model']]['raw_invalid_annotations']+=not metadata['annotation_valid']
        records.append(dict(cache_key=key,model=state['model'],response_id=response['id'],native_cost_usd=str(actual),annotation_valid=metadata['annotation_valid']))
    totals={model:dict(counts,usd=str(sum(Decimal(row['native_cost_usd']) for row in records if row['model']==model))) for model,counts in models.items()}
    result=dict(responses=len(records),settled_native_total_usd=str(exposure(states)),models=totals,records=records,invoice_verified=False,pending_attempts=0)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()

