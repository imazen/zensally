#!/usr/bin/env python3
"""Sequential explicit local training experiments; no distributed fleet jobs."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('python',type=Path);p.add_argument('plan',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    cases=json.loads(a.plan.read_text());seen=set();state={'supervisor_pid':os.getpid(),'completed':[],'required':len(cases)}
    for case in cases:
        if case['id'] in seen or not case['id'].replace('-','').isalnum():raise ValueError('distinct safe case IDs required')
        seen.add(case['id'])
    def update(stage,**values):
        state.update(stage=stage,checked_at=datetime.now(timezone.utc).isoformat(),**values)
        (a.output/(a.plan.stem+'-progress.json')).write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    for case in cases:
        out=a.output/case['id']
        if out.exists():raise ValueError('refuse to replace an attempted experiment')
        command=['/usr/bin/time','-v','-o',str(a.output/(case['id']+'-time.txt')),str(a.python),str(Path(__file__).with_name('train_probe.py')),str(out),*case['args']]
        update('starting',case=case['id'],command=command)
        with (a.output/(case['id']+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);update('running',child_pid=child.pid)
            deadline=time.monotonic()+600
            while child.poll() is None:
                if time.monotonic()>deadline:
                    child.terminate();child.wait();update('failed',error='600-second case deadline');raise RuntimeError('case exceeded deadline; preserved outputs')
                time.sleep(1)
            if child.returncode:
                update('failed',returncode=child.returncode);raise RuntimeError('case failed; inspect retained log before another experiment')
        result=json.loads((out/'result.json').read_text())
        if not result['all_gradients_finite'] or not result['updated_parameter_tensors']:raise ValueError('training suitability failed')
        state['completed'].append(case['id']);update('case-complete',result={k:result[k] for k in ['step_ms','images_per_second','cuda_peak_allocated','cuda_peak_reserved']})
    update('complete',child_pid=None)

if __name__=='__main__':main()
