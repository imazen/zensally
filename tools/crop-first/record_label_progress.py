#!/usr/bin/env python3
"""Resolve live label-audit workers and separate the newly authorized tranche."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess
from sunburst_generate import journal,exposure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('attention',type=Path);p.add_argument('progress',type=Path);a=p.parse_args()
    now=datetime.now(timezone.utc).isoformat();workers=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:args=(process/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(v==str(a.root) or v.startswith(str(a.root)+'/') for v in args) and any(Path(v).name in ['propose_crop_corridors.py','label_quality.py','batch_subject_zones.py'] for v in args):
            workers.append(dict(pid=int(process.name),producer=next(Path(v).name for v in args if Path(v).name in ['propose_crop_corridors.py','label_quality.py','batch_subject_zones.py'])))
    states=journal(a.root/'testing/budget.jsonl');settled=sum((Decimal(v['usd']) for v in states.values() if v['state']=='settled'),Decimal(0));pending=sum(v['state']=='reserved' for v in states.values())
    data=json.loads(a.progress.read_text());data.update(checked_utc=now,current_run=str(a.root),stage='nested crop label pilot active' if workers else 'inspecting nested crop label pilot',
        verified_training_pid=None,verified_live_label_workers=workers,no_new_paid_api_calls=not bool(states),
        current_observation=dict(checked_utc=now,stage='label proposals/reviews' if workers else 'no live label worker',workers=workers,pending_reservations=pending),
        next_action='Judge teacher-designed same-aspect maximum-area Minimal / tighter Maximal nested pairs and their intermediate pixels before admitting positive crop ranges or resuming fitting.')
    data['label_quality_budget']=dict(additional_authorization_usd='20',pilot_ceiling_usd='2',settled_usd=str(settled),pending_reservations=pending,exposure_usd=str(exposure(states)),
        prior20_dollar_study_settled_usd=data['additional_budget']['settled_total_usd'],combined_study_settled_usd=str(settled+Decimal(data['additional_budget']['settled_total_usd'])),
        authorization=str(a.root/'authorization.json'),ledger=str(a.root/'testing/budget.jsonl'),invoice_verified=False)
    data['frozen_packets'].update(real_attention_inputs='86aacbfb10ce299649029e4805efc73073fb542feb4b2f2401cc0d9cb8c32ef3',
        crop_label_quality_pilot='92838a0d3c83aa18b0a843bdbefc4087709d1350c2b10e662ca5563577938396')
    final=a.root/'analysis-phase1/summary.json'
    if final.exists():
        summary=json.loads(final.read_text());intervals=[r for r in summary['intervals'] if r['pair'].get('pair_protocol',1)==2]
        data['nested_crop_pilot']=dict(proposal_sources=summary['proposal_sources_v2'],proposed_pairs=summary['proposed_pairs_v2'],
            reviewed_intervals=len(intervals),all_five_frames_accepted=sum(r['all_reviewed_samples_accepted'] for r in intervals),
            excellent_endpoint_gate_passed=sum(r['eligible_for_weak_interval_admission'] for r in intervals),model_promoted=False,training_started=False)
        data['next_action']='Refine teacher-designed maximum-area Minimal / tighter same-aspect Maximal pairs using actual-pixel feedback and appropriate source/purpose coverage. Keep the zero excellent-interval baseline and all original quality/face guards; do not train from designer excellence flags alone.'
    if (a.attention/'cache/summary.json').exists():
        data['real_attention_cache']=dict(summary=json.loads((a.attention/'cache/summary.json').read_text()),registered_fits=15,completed_fits=0,
            fitting_waits_for_owner_directed_label_audit=True,root=str(a.attention),teacher_completion_verified=json.loads((a.attention/'teacher-completion-verification.json').read_text()))
    revision=subprocess.check_output(['git','rev-parse','origin/review/fixed-w24-depth'],text=True).strip();data['latest_pushed_review_commit']=revision
    runs=json.loads(subprocess.check_output(['gh','run','list','--commit',revision,'--limit','1','--json','databaseId'],text=True))
    data['latest_ci']=json.loads(subprocess.check_output(['gh','run','view',str(runs[0]['databaseId']),'--json','databaseId,headSha,status,conclusion'],text=True)) if runs else dict(headSha=revision,status='not listed')
    a.progress.rename(a.progress.with_name(a.progress.stem+'.before-label-'+now.replace(':','')+'.json'));a.progress.write_text(json.dumps(data,indent=2)+'\n')
    (a.root/'goal-observation.json').write_text(json.dumps(dict(checked_utc=now,workers=workers,budget=data['label_quality_budget'],registered_attention_fits_started=0),indent=2)+'\n')
    print(json.dumps(dict(stage=data['stage'],workers=workers,budget=data['label_quality_budget'])))


if __name__=='__main__':main()
