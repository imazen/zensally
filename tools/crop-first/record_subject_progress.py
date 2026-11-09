#!/usr/bin/env python3
"""Record actual funded-study ledger states and a verified worker PID."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path
from sunburst_generate import exposure,journal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('progress',type=Path);p.add_argument('--pid',type=int);a=p.parse_args()
    root=a.root;generation=root/('replacements' if (root/'replacements/budget.jsonl').exists() else 'expansion' if (root/'expansion/budget.jsonl').exists() else 'pilot')/'budget.jsonl'
    states=journal(generation);tests=journal(root/'testing/budget.jsonl')
    def settled(values):return sum((Decimal(v['usd']) for v in values.values() if v['state']=='settled'),Decimal(0))
    pid=None
    if a.pid:
        command=Path('/proc')/str(a.pid)/'cmdline'
        if not command.exists() or b'sunburst_generate.py' not in command.read_bytes():raise ValueError('specified generation PID does not match the actual worker')
        pid=a.pid
    now=datetime.now(timezone.utc).isoformat();data=json.loads(a.progress.read_text())
    a.progress.rename(a.progress.with_name(a.progress.name+'.bak-'+now.replace(':','')))
    data.update(checked_utc=now,current_run=str(root),stage='funded source generation' if pid else 'funded pilot analysis',
        current_producer='sunburst_generate.py' if pid else None,verified_training_pid=None,
        verified_pid_command_matches=bool(pid),current_observation=dict(pid=pid,checked_utc=now,stage='generating' if pid else 'no generation PID supplied'),
        finding='Broader photographic ranking improves the new cohort but regresses the old cohort; person-box auxiliary transfer gives no crop gain. Subject-choice supervision permits whole-face exclusion and rejects partial faces.',
        next_action='Complete native generation verification and lossless source publication; admit only audited crop references for controlled subject-choice fitting.')
    data['frozen_packets']['miap_broad_ranks']='21434d0f0ebbb88dc1ac0fa60b479af48e89d011cc03110a6e84427cc560a438'
    data['frozen_packets']['subject_choice_fits']='8818f1f01023e9552750e5965ee7e3374a42d571a944c7b9e913df1f470c1fa0'
    data['frozen_packets']['subject_choice_pilot']='5c4494e41afee6f79c3ca7394576b7e8e27978fdf72ceca2af6ea657aa7f4947'
    data['frozen_packets']['subject_choice_expansion']='98aa1cc81d13ee7aa1aa338bd6fd53a263e2dcd22d933f6a3b565d851f28ed7c'
    for name in ['miap-composition-transfer','miap-broad-ranks','subject-selection-v4']:
        path='/mnt/v/output/zensally/'+name+'/2026-11-09'
        if path not in data['evidence_roots']:data['evidence_roots'].append(path)
    data['additional_budget'].update(settled_generation_usd=str(settled(states)),settled_testing_usd=str(settled(tests)),
        settled_total_usd=str(settled(states)+settled(tests)),pending_generation_reservations=sum(v['state']=='reserved' for v in states.values()),
        pending_testing_reservations=sum(v['state']=='reserved' for v in tests.values()),
        total_exposure_usd=str(exposure(states)+exposure(tests)),active_generation_ledger=str(generation))
    data['funded_subject_pilot']=dict(generated=4,terra_responses=4,sol_responses=1,
        raw_invalid_terra_responses=1,explicit_coordinate_preserving_alias_repairs=1,proposed_crop_options=38,
        supplied_face_geometry_rejections=1,complete_face_coverage_certified=False,
        native_png_bytes=4140416,lossless_jxl_bytes=2277118,lossless_exact_readbacks=4,
        source_packet_nas_frozen=True)
    observations={};live={}
    for folder,producer in [('real-terra','batch_subject_zones.py'),('real-sol','batch_subject_zones.py'),('expansion-terra','batch_subject_zones.py'),('sol-primary','batch_subject_zones.py'),('fits','fit_subject_sets.py'),('replay-control-fits','fit_subject_sets.py')]:
        path=root/folder/'PROGRESS.json'
        if not path.exists():continue
        record=json.loads(path.read_text());worker=record.get('pid');command=Path('/proc')/str(worker)/'cmdline'
        running=worker is not None and command.exists() and producer.encode() in command.read_bytes()
        observations[folder]={k:v for k,v in record.items() if k!='results'}
        observations[folder]['verified_pid_command_matches']=running
        if running:live[folder]=worker
        elif worker is not None:observations[folder]['failure']='recorded worker no longer exists; inspect retained artifacts'
    data.update(funded_study_observations=observations,verified_live_funded_workers=live,
        stage='funded references/training active' if live else 'funded references and15controlled fits complete; analyzing and improving candidate coverage')
    data['completed_required_fits']['subject_sets']=[observations.get('fits',{}).get('completed',0),12]
    data['completed_required_fits']['subject_replay_control']=[observations.get('replay-control-fits',{}).get('completed',0),3]
    if (root/'dataset-88-remote-verification.json').exists():data['funded_dataset_publication']=json.loads((root/'dataset-88-remote-verification.json').read_text())|{'records':'retained in source verification artifact'}
    if (root/'analysis/summary.json').exists():
        analysis=json.loads((root/'analysis/summary.json').read_text())
        data['subject_set_finding']={k:analysis[k] for k in ['averages','paired','new_development_families','model_promoted']}
        data['next_action']='Improve crop-bank coverage before another architecture fit; preserve independent teacher disagreements and closed face-coverage guards.'
    data['no_new_paid_api_calls']=False
    data['full_live_study_packet_nas_frozen']=False
    if observations.get('sol-primary',{}).get('stage')!='complete':data['next_action']='Complete tracked Sol-primary references with cached replay, audit geometry and compare matched fits; preserve the frozen Terra study and improve bank coverage independently.'
    a.progress.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(stage=data['stage'],pid=pid,budget=data['additional_budget'])))

if __name__=='__main__':main()

