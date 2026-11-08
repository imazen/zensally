#!/usr/bin/env python3
"""Summarize complete retained timing packets, without treating frames as replicates."""
import argparse,gzip,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();out={}
for study in ['reference','rust','face-followup','pair-followup','native-graph']:
 packet=a.root/(study+'-timing')/'packet.json.gz';d=json.loads(gzip.decompress(packet.read_bytes()))
 if not d['complete']:raise ValueError('incomplete benchmark cannot supply model means')
 arms=[d['plan']['baseline']]+d['plan']['candidates']+[{'id':'__aa'}];groups={}
 for inv in d['invocations']:
  if inv['status']!='completed' or len(inv['values_ns'])!=1:raise ValueError('failed or malformed timing')
  groups.setdefault(arms[inv['arm']]['id'],[]).append(inv['values_ns'][0]/1e6)
 out[study]={'placement':d['placement'],'inference':'exploratory; background load observed; different outputs unchecked',
  'timing_scope':d['plan']['warmup'],'models':{k:{'processes':len(v),'mean_ms':statistics.mean(v),'min_ms':min(v),'max_ms':max(v)} for k,v in groups.items()}}
with (a.root/'timing-summary.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v['models'] for k,v in out.items()},indent=2))
