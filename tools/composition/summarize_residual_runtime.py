#!/usr/bin/env python3
"""Summarize native timings, embedded payloads and measured allocation counts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from statistics import mean


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('target',type=Path);a=p.parse_args();root=a.root
    timing=[]
    for case,count in [(0,1),(1,22)]:
        records=[r for path in (root/'timing').glob('case-'+str(case)+'-block-*.json') for r in json.loads(path.read_text())]
        for arm,label in [(0,'baseline'),(1,'residual'),(2,'aa')]:
            selected=[r for r in records if r['arm']==arm]
            if len(selected)!=4 or any(r['status']!='completed' for r in selected):raise ValueError('all four paired blocks required')
            values=[value/1000 for r in selected for value in r['values_ns']]
            timing.append(dict(queries_per_cycle=count,arm=label,process_means_us_per_query=values,mean_us_per_query=mean(values),measured_cycle_us=mean(values)*count))
    allocations={}
    for name in ['short','long']:
        content=(root/('heap-'+name)/'report.txt').read_text();matches=re.findall(r'^calls to allocation functions: (\d+)',content,re.M)
        if len(matches)!=1:raise ValueError('allocation summary absent or ambiguous')
        allocations[name]=int(matches[0])
    binary=(root/'binaries/residual-head').read_bytes();embedded={};destination=root/'embedded-models';destination.mkdir(exist_ok=False)
    for path in a.target.glob('release/build/zensally-residual-head-study-*/out/*.znpr'):
        raw=path.read_bytes()
        if raw not in binary:continue
        name=path.stem;digest=hashlib.sha256(raw).hexdigest()
        if name in embedded and embedded[name]['sha256']!=digest:raise ValueError('ambiguous embedded model')
        if name not in embedded:shutil.copy2(path,destination/path.name)
        embedded[name]=dict(bytes=len(raw),sha256=digest,literal_payload_verified_in_executable=True)
    if set(embedded)!={'base','residual'}:raise ValueError('both compiled payloads required')
    result=dict(timing=timing,allocation_calls=allocations,embedded_models=embedded,binary_bytes=len(binary),api_calls=0,test_inference=False,model_promoted=False,
        scope='Native dense head only; cached96-value candidate inputs. Whole encoder/pooling/first-scorer pipeline excluded. Exploratory four-block pair, not a promotion or clean-environment claim.')
    with (root/'summary.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
