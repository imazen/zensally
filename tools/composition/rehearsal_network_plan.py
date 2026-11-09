#!/usr/bin/env python3
"""Paired complete-neural reference cost; no TEST or inference API calls."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['python','old','new','export']:p.add_argument(key,type=Path)
    a=p.parse_args();runner=Path(__file__).with_name('bench_rehearsal_network.py').resolve();root=a.export.resolve();qualification=json.loads((root/'qualification.json').read_text())
    if any(r['gate_violations'] for r in qualification['totals'].values()):raise ValueError('neural gate required')
    def arm(name):return dict(id=name,program='/usr/bin/env',args=[str(a.python),str(runner),str(a.old),str(a.new),str(root),'--arm',name,'--phase','${phase}'],env=dict(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),revision='qualified fixed encoder and query graphs',timeout_seconds=120)
    inputs=dict(old=str(a.old),new=str(a.new),runner=str(runner),qualification=str(root/'qualification.json'),**{name:str(root/(name+'.onnx')) for name in ['encoder','baseline-query','candidate-query']})
    plan=dict(schema=1,baseline=arm('baseline'),candidates=[arm('candidate')],cases=[dict(id=phase,fields=dict(phase=phase),inputs=inputs) for phase in ['one','bank']],metric=dict(extract=dict(kind='json',prefix='RESULT ',pointer='/neural_ms'),unit='ms',expected=1,divide_by=1),check=dict(kind='json',prefix='CHECK ',pointers=['/sources','/requests','/gate_violations']),warmup='all49 old/new DEV images, one full phase sweep excluded; retained encoder/query bindings and preallocated score/mask buffers; three measured sweeps; one encoder per image',blocks=4,aa=True,deadline_seconds=900)
    with (root/'timing-plan.json').open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
