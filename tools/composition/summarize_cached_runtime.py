#!/usr/bin/env python3
"""Read retained paired packets and report measured process means by arm."""
import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('zenbench',type=Path)
    parser.add_argument('--studies',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summaries={}
    for name in args.studies:
        packet=args.root/name/'packet.json.gz'
        result=subprocess.run([str(args.zenbench),'read',str(packet),'--format','json'],
                              capture_output=True,text=True,check=True)
        data=json.loads(result.stdout)
        if not data['complete']:raise ValueError('incomplete paired study')
        baseline=data['plan']['baseline']['id']
        names=[baseline]+[a['id'] for a in data['plan']['candidates']]+['__aa']
        values=defaultdict(list)
        for run in data['invocations']:
            if run['status']!='completed' or len(run['values_ns'])!=1:
                raise ValueError('failed or ambiguous timing invocation')
            values[names[run['arm']]].append(run['values_ns'][0]/1e6)
        summaries[name]=dict(fingerprint=data['fingerprint'],baseline=baseline,
            invocations=len(data['invocations']),placement=data['placement'],
            mean_ms_per_source={arm:statistics.mean(samples) for arm,samples in values.items()},
            samples_ms_per_source=dict(values),paired=data['summaries'],
            cases=[case['id'] for case in data['plan']['cases']],
            scope=data['plan']['warmup'],
            confirmatory=False)
        with (args.root/(name+'-readable.json')).open('x') as stream:
            stream.write(result.stdout)
    with args.output.open('x') as stream:
        json.dump(summaries,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:dict(means=v['mean_ms_per_source'],invocations=v['invocations']) for k,v in summaries.items()},indent=2))


if __name__=='__main__':main()
