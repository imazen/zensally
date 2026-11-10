#!/usr/bin/env python3
"""Read-only verification of complete native generation, cached usage and budget."""
import argparse,json
from decimal import Decimal
from pathlib import Path
from PIL import Image
from sunburst_generate import budget_approval,canonical,decode,exposure,journal,payload,sha,usage_cost
from sunburst_costs import output_tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--carry-journal',type=Path)
    p.add_argument('--authorization',type=Path)
    a=p.parse_args();jobs=json.loads((a.directory/'plan.json').read_text());rows=[];keys=set()
    for job in jobs:
        key=sha(canonical(payload(job)));keys.add(key)
        png,response,meta=decode(a.cache/key,payload(job))
        record=json.loads((a.directory/(job['id']+'.json')).read_text())
        source=a.directory/(job['id']+'.png')
        if source.read_bytes()!=png or sha(png)!=record['png_sha256']:raise ValueError('native image differs from cache')
        with Image.open(source) as image:
            image.load()
            if image.mode!='RGB' or image.size!=tuple(map(int,job['size'].split('x'))):raise ValueError('native dimensions/mode differ')
        cost=usage_cost(response['usage'])
        if cost!=Decimal(record['published_rate_usd']) or cost!=Decimal(meta['published_rate_usd']):raise ValueError('native price differs')
        if record['usage']!=response['usage'] or record['key']!=key:raise ValueError('native usage/key differs')
        if response['usage']['output_tokens']!=output_tokens(job['size']):raise ValueError('native output tokens differ from calculator')
        rows.append({'id':job['id'],'key':key,'png_sha256':sha(png),'usd':str(cost),'usage':response['usage']})
    states=journal(a.directory/'budget.jsonl')
    config_path=a.directory/'budget-config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {'ceiling_usd':'.95'}
    carry=Decimal(config.get('carry_usd','0'))
    if a.carry_journal:
        if sha(a.carry_journal.read_bytes())!=config['carry_sha256'] or exposure(journal(a.carry_journal))!=carry:raise ValueError('carry source changed')
        keys.add('carry-'+config['carry_sha256'])
    elif carry:raise ValueError('carry source required for complete accounting verification')
    if set(states)!=keys or any(v['state']!='settled' for v in states.values()):raise ValueError('incomplete or unexpected paid attempts')
    total=sum((Decimal(r['usd']) for r in rows),Decimal(0))
    limit,approval=budget_approval(a.authorization)
    if any(config.get(key)!=value for key,value in approval.items()):raise ValueError('budget authorization binding differs')
    if exposure(states)!=total+carry or total+carry>=Decimal(config['ceiling_usd']) or Decimal(config['ceiling_usd'])>limit:raise ValueError('budget accounting differs')
    result={'images':len(rows),'native_pngs_checked':len(rows),'cached_responses_checked':len(rows),'new_published_rate_usd':str(total),'carried_published_rate_usd':str(carry),'total_published_rate_usd':str(total+carry),'pending_attempts':0,'invoice_verified':False,'records':rows}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
