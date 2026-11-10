#!/usr/bin/env python3
"""Record actual MIAP process/artifact progress without a background timer."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('producer');a=p.parse_args()
    observation=json.loads((a.run/'progress.json').read_text());pid=observation.get('pid');alive=False
    if pid:
        try:
            os.kill(pid,0)
            arguments=Path('/proc')/str(pid)/'cmdline'
            alive=a.producer in arguments.read_text().replace('\x00',' ')
        except (ProcessLookupError,FileNotFoundError):pass
        if not alive:raise ValueError('reported PID is absent or belongs to another producer; inspect retained failure logs')
    target=Path('/mnt/v/output/zensally/composition-goal-progress.json');current=json.loads(target.read_text());now=datetime.now(timezone.utc)
    target.with_name('composition-goal-progress-before-'+now.strftime('%Y%m%dT%H%M%S%f')+'.json').write_bytes(target.read_bytes())
    current.update(checked_utc=now.isoformat(),stage='MIAP '+a.run.parent.name+' '+observation.get('stage','unknown'),
        verified_training_pid=pid if alive else None,current_run=str(a.run),current_epoch=observation.get('epoch'),current_arm=observation.get('arm'),
        current_completed_fits=observation.get('completed_fits',0),current_required_fits=observation.get('required_fits',0),
        current_observation=observation,current_producer=a.producer,verified_pid_command_matches=alive)
    target.write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps(dict(run=str(a.run),observation=observation,pid_verified=alive)))


if __name__=='__main__':main()
