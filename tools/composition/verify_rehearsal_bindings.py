#!/usr/bin/env python3
"""Check actual timing runner I/O bindings against unbound graph execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import bench_rehearsal_network as bench


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['old','new','export']:p.add_argument(key,type=Path)
    a=p.parse_args();original_session=bench.session;original_binding=bench.binding;records={};counts=dict(calls=0,arrays=0)
    class CheckedSession:
        def __init__(self,*args):self.real=original_session(*args)
        def __getattr__(self,name):return getattr(self.real,name)
        def run_with_iobinding(self,binding):
            inputs,outputs=records[id(binding)];self.real.run_with_iobinding(binding)
            expected=self.real.run(None,inputs)
            for spec,reference in zip(self.real.get_outputs(),expected):
                actual=outputs[spec.name]
                if not np.array_equal(actual,reference) or not np.isfinite(actual).all():raise ValueError('bound and unbound graph outputs differ')
                counts['arrays']+=1
            counts['calls']+=1
    def checked_binding(graph,inputs,outputs):
        binding=original_binding(graph.real,inputs,outputs);records[id(binding)]=(inputs,outputs);return binding
    bench.session=CheckedSession;bench.binding=checked_binding
    for arm in ['baseline','candidate']:
        records.clear();sys.argv=['verify',str(a.old),str(a.new),str(a.export),'--arm',arm,'--phase','bank','--repeats','1'];bench.main()
    if counts!=dict(calls=4508,arrays=4704):raise ValueError('complete two-arm49-source binding coverage required')
    result=dict(counts,arms=2,sources=49,queries_per_source=22,all_bound_arrays_exact=True,api_calls=0,test_inference=False,limitation='Instrumented correctness run. Its RESULT timing lines include independent reference calls and are not performance evidence.')
    with (a.export/'binding-check.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
