#!/usr/bin/env python3
"""Run independent reference attempts; retain every failure without retries."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--rust',type=Path);p.add_argument('--models',nargs='+');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);doc=json.loads(a.manifest.read_text());rows=[]
models=['ultraface','front','blazeface320','yunet','selfie','microsalnet','u2netp'] if a.rust else [r['id'] for r in doc['models']]
if a.models:
 if not set(a.models)<=set(models):raise ValueError("unknown requested models")
 models=a.models
for model in models:
 command=[str(a.rust.resolve()),'evaluate',model,str(a.manifest.parent/'inputs.tsv'),str(a.output/model),'1'] if a.rust else [sys.executable,str(Path(__file__).with_name('mediapipe_probe.py')),'evaluate',str(a.manifest),model,str(a.output/model)]
 with (a.output/(model+'.log')).open('x') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 row={'model':model,'returncode':result.returncode,'command':command};rows.append(row)
 (a.output/'progress.json').write_text(json.dumps({'completed':rows,'required':len(models),'running':False},indent=2)+'\n');print(json.dumps(row),flush=True)
(a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if any(r['returncode'] for r in rows):sys.exit(1)
