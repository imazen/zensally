#!/usr/bin/env python3
"""Strict scalar summary of two qualified complete-neural paired studies."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
from statistics import mean


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('gather',type=Path);a=p.parse_args();rows=[];qualifications=[]
    for label,root in [('raw',a.raw),('gather',a.gather)]:
        qualification=json.loads((root/'qualification.json').read_text());qualifications.append(qualification)
        if qualification['test_inference'] or any(r['gate_violations'] for r in qualification['totals'].values()):raise ValueError('qualified FIT/DEV-only graphs required')
        bindings=json.loads((root/'binding-check.json').read_text())
        if not bindings['all_bound_arrays_exact']:raise ValueError('exact binding check required')
        timing=[]
        for case,phase in [(0,'one'),(1,'bank')]:
            records=[r for path in (root/'timing').glob('case-'+str(case)+'-block-*.json') for r in json.loads(path.read_text())]
            for arm in [0,1,2]:
                values=[v/1e6 for r in records if r['arm']==arm and r['status']=='completed' for v in r['values_ns']]
                if len(values)!=4:raise ValueError('four complete paired blocks required')
                timing.append(dict(phase=phase,arm=arm,process_means_ms=values,mean_ms=mean(values)))
        counters=(root/'network.perf-stat').read_text();rss=(root/'network.time').read_text()
        memory=re.search(r'([\d.]+)\s+%\s+tma_memory_bound',counters);core=re.search(r'([\d.]+)\s+%\s+tma_core_bound',counters);maximum=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',rss)
        if not memory or not core or not maximum:raise ValueError('complete hardware/RSS evidence required')
        rows.append(dict(study=label,timing=timing,graphs=qualification['exports'],gate=qualification['totals'],binding_arrays=bindings['arrays'],memory_bound_percent=float(memory[1]),core_bound_percent=float(core[1]),profiling_max_rss_kib=int(maximum[1])))
    if qualifications[0]['inputs']!=qualifications[1]['inputs']:raise ValueError('same student and caches required')
    result=dict(studies=rows,software={k:importlib.metadata.version(k) for k in ['torch','onnx','onnxruntime','numpy']},api_calls=0,test_inference=False,model_promoted=False,scope='Full neural reference path on cached256x256 tensors; excludes decoding/resizing/candidate generation/face guards. ORT, not complete native Rust deployment.')
    with (a.gather/'summary.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
