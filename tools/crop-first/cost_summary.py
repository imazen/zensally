#!/usr/bin/env python3
"""Reconcile recorded session usage by explicit task scope; never an invoice."""
import argparse,json
from decimal import Decimal
from pathlib import Path
from cli_records import read_records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('rollout',type=Path);a=p.parse_args()
    ledgers={name:json.loads((a.root/'costs'/f'{name}.json').read_text()) for name in ['terra-full','astra','token-large','token-medium','token-small']}
    scopes={};current='earlier_teacher_work';failed=[]
    for row in read_records(a.rollout):
        v=row.get('payload',{})
        if row['type']=='response_item' and v.get('role')=='user':
            for c in v.get('content',[]):
                text=c.get('text','')
                if text.startswith('New bounded vision-'):
                    current='coverage_audit' if '/coverage-audit/batches/' in text else 'expanded_main' if '/zensally-crop-expanded-20261107/batches/' in text else 'earlier_teacher_work'
        if row['type']=='event_msg' and v.get('type')=='task_complete':
            scopes[row['timestamp']]=current
            if v.get('error'):failed.append({'completed_at':row['timestamp'],'scope':current,'error':v['error']})
    totals={};turns={}
    for t in ledgers['terra-full']['turns']:
        scope=scopes[t['completed_at']];totals[scope]=totals.get(scope,Decimal(0))+Decimal(t['standard_api_equivalent_usd']);turns[scope]=turns.get(scope,0)+1
    result={'rates_verified':'2026-11-08','actual_billed_amount_known':False,'api_invoice_available':False,'scope':'named teacher, Astra and matched preview-probe sessions; excludes coordinator and other work','teacher_scopes':{k:{'completed_turns_including_failures':turns[k],'recorded_standard_api_equivalent_usd':str(v)} for k,v in totals.items()},'sessions':{k:{'model':v['model'],'recorded_standard_api_equivalent_usd':v['standard_api_equivalent_usd'],'total_usage':v['total_usage']} for k,v in ledgers.items()},'combined_recorded_standard_api_equivalent_usd':str(sum(Decimal(v['standard_api_equivalent_usd']) for v in ledgers.values())),'failed_turns':failed,'note':'Standard published API comparison with per-request long-context adjustments and recorded cache hits. Unreported usage is unknown, not assumed free. No subscription credit charge, allowance or future-run estimate.'}
    (a.root/'cost-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
