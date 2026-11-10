#!/usr/bin/env python3
"""Execute an explicit finite list of two-source batches through visible Herdr.

Stops on any model, schema, file or process failure. No retries, extra sources,
model changes, training admission or hidden jobs. Supervision records every stage.
"""
import argparse,hashlib,json,os,shlex,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
SESSION='01a1b826-36c2-7922-a393-ed3007317e57'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('rollout',type=Path);p.add_argument('batches');p.add_argument('--execute',action='store_true');a=p.parse_args()
    names=a.batches.split(',')
    if len(set(names))!=len(names) or any(len(n)!=2 or not n.isdigit() or not 0<=int(n)<16 for n in names):raise ValueError('explicit distinct batch IDs 00..15 required')
    state={'pid':os.getpid(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'batches':names,'teacher_agent':'crop-judge-terra','teacher_session':SESSION,'completed_batches':[]}
    def checkpoint(stage,name=None,error=None):
        state.update(stage=stage,batch=name,checked_at=datetime.now(timezone.utc).isoformat(),error=error)
        (a.root/'supervision.json').write_text(json.dumps(state,indent=2)+'\n')
        lock=REPO/'.workongoing'
        if lock.read_text().split()[1]!='codex-root':raise RuntimeError('checkout lock owner changed')
        lock.write_text(state['checked_at']+' codex-root expanded study '+stage+' '+str(name)+'\n')
        print(json.dumps(state),flush=True)
    def run(args):return subprocess.run(args,cwd=REPO,check=True)
    def just(*args):return run(['just',*map(str,args)])
    def agent():
        v=json.loads(subprocess.run(['herdr','agent','get','crop-judge-terra'],check=True,capture_output=True,text=True).stdout)['result']['agent']
        if v['agent_session']['value']!=SESSION:raise RuntimeError('teacher session changed')
        return v
    def request(kind,batch):
        before=agent();seq=before.get('completion_seq',-1)
        if before['agent_status'] not in ['idle','done']:raise RuntimeError('teacher is already active; do not queue')
        just('crop-expanded-send',kind,batch)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            v=agent()
            if v['agent_status'] in ['blocked','unknown']:raise RuntimeError('teacher needs inspection: '+v['agent_status'])
            if v.get('completion_seq',-1)>seq and v['agent_status'] in ['idle','done']:return
            time.sleep(1)
        raise RuntimeError('teacher turn outstanding after 300s; inspect it, do not resubmit')
    remote='i270';remote_root='/mnt/v/output/zensally/crop-first/2026-11-07-expanded';remote_inputs='/mnt/v/input/zensally-crop-expanded-2026-11-07';source='/home/lilith/tmp/zensally-memory-20261106/source';target='/home/lilith/tmp/zensally-tiling-20261106/target';python='/home/lilith/tmp/zensally-aspect-python-20261107/bin/python3'
    def upload(path):run(['scp','-q',str(path),f'{remote}:{remote_root}/batches/'])
    def remote_command(args,log=None):
        command=f'cd {shlex.quote(source)} && export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" TMPDIR=/home/lilith/tmp/zensally-tiling-20261106/scratch ZENSALLY_PREVIEW_RESIZER={target}/release/zensally-crop-preview && '+shlex.join(args)
        if log:
            with log.open('x') as f:subprocess.run(['ssh',remote,command],stdout=f,stderr=subprocess.STDOUT,check=True)
        else:run(['ssh',remote,command])
    plan=[{'batch':n,'proposal_call':not (a.root/f'batches/{n}-proposals.json').exists(),'judge_call':not (a.root/f'batches/{n}-labels.json').exists()} for n in names]
    print(json.dumps({'explicit_plan':plan,'model_calls':sum(r['proposal_call']+r['judge_call'] for r in plan)}),flush=True)
    if not a.execute:return
    if os.environ.get('HERDR_ENV')!='1':raise RuntimeError('Herdr-managed environment required')
    try:
        for n in names:
            if (a.root/f'batches/{n}-labels.json').exists():raise ValueError('explicit execution refuses to replace completed labels')
            source_batch=a.root/f'batches/{n}-sources-local.json';proposals=a.root/f'batches/{n}-proposals.json'
            if not proposals.exists():
                checkpoint('proposing',n);request('proposal',source_batch)
                just('crop-expanded-capture',a.rollout,source_batch,proposals)
            upload(proposals);upload(proposals.with_suffix('.evidence.json'))
            if not (a.root/f'batches/{n}-tasks.json').exists():
                probe=subprocess.run(['ssh',remote,'test -f '+shlex.quote(remote_root+f'/batches/{n}-tasks.json')])
                if probe.returncode not in [0,1]:raise RuntimeError('cannot inspect remote render state')
                if probe.returncode==1:
                    checkpoint('rendering',n)
                    remote_command(['/home/lilith/work/claudehints/scripts/run-heavy','--mem','4G','--jobs','1','--','just','crop-expanded-render-many',python,remote_inputs,remote_root,target+'/release/examples/crop_label_pairs',n],a.root/f'batches/{n}-render.log')
            just('crop-expanded-fetch',source_batch,a.root,remote,remote_root)
            checkpoint('judging',n);request('judge',a.root/f'batches/{n}-tasks-judge.json')
            labels=a.root/f'batches/{n}-labels.json';just('crop-first-cli-capture',a.rollout,a.root/f'batches/{n}-tasks-local.json',labels);upload(labels)
            state['completed_batches'].append(n);checkpoint('batch-complete',n)
            remote_command(['python3','tools/crop-first/expanded.py','status',remote_inputs,remote_root])
        checkpoint('complete')
    except BaseException as e:
        checkpoint('failed',state.get('batch'),str(e));raise


if __name__=='__main__':main()
