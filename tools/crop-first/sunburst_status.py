#!/usr/bin/env python3
"""Resolve a generation run from its ledger, source files and live worker processes."""
import argparse,datetime,hashlib,json
from decimal import Decimal
from pathlib import Path
from sunburst_generate import journal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--worker',type=Path,required=True)
    p.add_argument('--build-commit',required=True);p.add_argument('--write',action='store_true')
    a=p.parse_args();jobs=json.loads((a.directory/'plan.json').read_text())
    states=journal(a.directory/'budget.jsonl');pids=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/'cmdline').read_bytes().split(b'\0')
            positions=[i for i,arg in enumerate(args) if arg==str(a.worker).encode()]
            if any(i>0 and args[i-1].endswith(b'python3') for i in positions):pids.append(int(entry.name))
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    completed=[j['id'] for j in jobs if (a.directory/(j['id']+'.png')).exists() and (a.directory/(j['id']+'.json')).exists()]
    pending=sum(v['state']=='reserved' for v in states.values())
    done=len(completed)==len(jobs) and not pending
    stage='generated' if done else 'generating' if pids else 'failed-or-interrupted'
    report={'run_directory':str(a.directory),'build_commit':a.build_commit,'worker_sha256':hashlib.sha256(a.worker.read_bytes()).hexdigest(),'stage':stage,'completed':len(completed),'required':len(jobs),'verified_worker_chain_pids':pids,'last_check_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'settled_usd_including_prior':str(sum((Decimal(v['usd']) for v in states.values() if v['state']=='settled'),Decimal(0))),'pending_reservations':pending,'failure':None if done or pids else 'worker chain absent before completion; inspect batch.log and retained reservations','next_action':'validate native source hashes/usage, review source content, lossless JXL publication'}
    if a.write:(a.directory/'PROGRESS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if stage=='failed-or-interrupted':raise SystemExit(1)


if __name__=='__main__':main()
