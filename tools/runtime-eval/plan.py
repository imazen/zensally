#!/usr/bin/env python3
"""Declare an exploratory zenbench comparison; models need not agree semantically."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--python',required=True);p.add_argument('--rust',type=Path);a=p.parse_args()
doc=json.loads(a.manifest.read_text());script=Path(__file__).with_name('mediapipe_probe.py').resolve()
models=['ultraface','front','blazeface320','yunet','selfie','microsalnet','u2netp'] if a.rust else [r['id'] for r in doc['models']]
inputs={'manifest':str(a.manifest.resolve())}
if a.rust:inputs['inputs.tsv']=str(a.manifest.parent/'inputs.tsv')
else:inputs['runner']=str(script)
for r in doc['images']:
 if r['benchmark']:inputs[r['id']]=r['rgb']
if not a.rust:
 for r in doc['models']:inputs[r['id']]=r['path']
def arm(model):
 args=['bench',model,'${inputs.tsv}','${output}/work','20'] if a.rust else [str(script),'bench','${manifest}',model,'${output}/work','--repeats','20']
 return {'id':model,'program':str(a.rust.resolve()) if a.rust else '/usr/bin/env','args':args if a.rust else [a.python]+args,'env':{'TMPDIR':'/home/lilith/tmp','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'},'revision':'runtime-eval snapshot; packet hashes identify actual inputs','timeout_seconds':120}
plan={'schema':1,'baseline':arm(models[0]),'candidates':[arm(m) for m in models[1:]],'cases':[{'id':'six-native-photos','inputs':inputs}], 'metric':{'extract':{'kind':'json','prefix':'RESULT ','pointer':'/inference_ns'},'unit':'ns','expected':1,'divide_by':1},'check':{'kind':'unchecked','reason':'Different pretrained models/tasks intentionally produce different outputs. Quality is inspected separately on the same frozen corpus; this is not a correctness-equivalence experiment.'},'warmup':'One full sweep of six preloaded native-RGB images per model and fresh process, then 20 sweeps; includes Task/adapter preprocessing, inference and returned postprocessing, excludes source load/hash, model load, image construction and IO. Rust ImageRef creation is included. Mean per image, not p95.','blocks':4,'aa':True,'deadline_seconds':1200}
with a.output.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
