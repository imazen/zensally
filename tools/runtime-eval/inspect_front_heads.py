#!/usr/bin/env python3
"""Read-only head layout proof using the project's existing ONNX field reader."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_microsalnet import fields
model=gzip.decompress((Path(__file__).resolve().parents[2]/'crates/zensally-tract/models/face_detection_front_128x128_float32.onnx.gz').read_bytes())
graph=next(v for k,v in fields(model) if k==7);weights={};nodes=[]
for k,v in fields(graph):
 if k==5:
  items=list(fields(v));name=next(x.decode() for t,x in items if t==8);weights[name]=[x for t,x in items if t==1]
 if k==1:
  items=list(fields(v));op=next(x.decode() for t,x in items if t==4);inputs=[x.decode() for t,x in items if t==1];outputs=[x.decode() for t,x in items if t==2];attrs={}
  for t,x in items:
   if t==5:
    a=list(fields(x));name=next(b.decode() for j,b in a if j==1);attrs[name]=[b for j,b in a if j==8]
  nodes.append({'op':op,'inputs':inputs,'outputs':outputs,'attributes_ints':attrs})
heads=[]
for n in nodes:
 if n['op']=='Conv' and any('/conv2d_'+str(i)+'/' in n['outputs'][0] for i in range(17,21)):
  trans=next(x for x in nodes if x['op']=='Transpose' and n['outputs'][0] in x['inputs']);reshape=next(x for x in nodes if x['op']=='Reshape' and trans['outputs'][0] in x['inputs']);heads.append({'conv_output':n['outputs'][0],'weight_dims':weights[n['inputs'][1]],'transpose_permutation':trans['attributes_ints']['perm'],'output':reshape['outputs'][0]})
print(json.dumps({'heads':heads,'interpretation':'NCHW convolution outputs transpose [0,2,3,1] to NHWC before flattening. Heads with 6 score or 96 box channels have 6 anchors per cell; 2/32-channel heads have 2.'},indent=2))
