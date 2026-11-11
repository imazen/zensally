#!/usr/bin/env python3
"""Zenbench external worker: native Mac timing excludes SSH and evidence transfer."""
import argparse
from pathlib import Path
import shlex
import subprocess
import uuid

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('arm',choices=['stock','single']);p.add_argument('output',type=Path)
p.add_argument('--host',required=True);p.add_argument('--via',required=True)
p.add_argument('--root',required=True);p.add_argument('--wrapper',required=True)
a=p.parse_args()
remote=a.root
dest=remote+'/invocations/'+uuid.uuid4().hex
def run(args,**kwargs):
    command=shlex.join(['ssh','-o','BatchMode=yes',a.host,shlex.join(args)])
    return subprocess.run(['ssh','-o','BatchMode=yes',a.via,command],check=True,**kwargs)
run([a.wrapper,'--mem','2G','--jobs','1','--',
     remote+'/build/MinSizeRel/ort-native-'+a.arm,dest],timeout=90)
data=run(['cat',dest+'/outputs.f32'],capture_output=True,timeout=30).stdout
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'outputs.f32').write_bytes(data)
