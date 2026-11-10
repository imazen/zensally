#!/usr/bin/env python3
"""Freeze the five-view/reference and corrected native-face followup."""
import argparse,copy,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('binary',type=Path);p.add_argument('--pairs',action='store_true');a=p.parse_args()
plan=json.loads((a.root/'reference-plan.json').read_text());sparse=next(x for x in plan['candidates'] if x['id']=='float16-blaze_face_full_range_sparse');tile=copy.deepcopy(sparse);tile['id']='sparse-tiles5';tile['args'][1]=str(Path(__file__).with_name('tile_probe.py').resolve())
fixed={'id':'front-fixed','program':str(a.binary.resolve()),'args':['bench','front',str(a.root/'inputs.tsv'),'${output}/work','20'],'env':sparse['env'],'revision':'d05dd502 corrected anchor order','timeout_seconds':120}
plan['baseline']=sparse;plan['candidates']=[tile,fixed];plan['cases'][0]['inputs'].update({'tile-runner':tile['args'][1],'native-inputs':str(a.root/'inputs.tsv')})
plan['warmup']+=' Five-view arm includes global plus four overlapping 75% width/height views, view copies, coordinate remapping and greedy NMS at IoU 0.3. Different models/guards are intentionally not output-equivalent.'
if a.pairs:
 def pair(name):
  arm=copy.deepcopy(fixed);arm['id']=name;arm['args'][1]=name;arm['revision']='same runtime-probe binary, serial components';return arm
 plan['baseline']=pair('ultraface');plan['candidates']=[pair('ultra-selfie'),pair('ultra-microsalnet')];plan['warmup']='One sweep of six native RGB images, then 20 sweeps. Serial component pair includes both complete adapter calls and returned results; no pooling or cross-model preprocessing reuse. Input/model loads and IO excluded.'
with (a.root/('pair-followup-plan.json' if a.pairs else 'face-followup-plan.json')).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
