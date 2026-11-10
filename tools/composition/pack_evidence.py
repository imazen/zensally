#!/usr/bin/env python3
"""Snapshot completed evidence outside Git, dereferencing links without deleting inputs."""
import argparse,hashlib,importlib.metadata,json,os,shutil,subprocess
from pathlib import Path

def copy_file(src,dst):
 # Real files in the packet: no links back to mutable source directories.
 shutil.copy2(src,dst,follow_symlinks=True)
 return dst
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first_gate',type=Path);p.add_argument('auxiliary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for name,root in [('first-gate',a.first_gate),('auxiliary',a.auxiliary)]:
  shutil.copytree(root,a.output/name,copy_function=copy_file,ignore=shutil.ignore_patterns('GOAL_PROGRESS.json','.cache','__pycache__'))
 logs=a.output/'logs';logs.mkdir()
 for f in sorted(Path('/home/lilith/tmp').glob('zensally-composition-*20261108.log')):copy_file(f,logs/f.name)
 source=a.output/'source';source.mkdir();files=list(Path('tools/composition').glob('*'))+list(Path('training').glob('composition_*.py'))+list(Path('training').glob('test_composition_*.py'))+[Path('docs/experiments/COMPOSITION_FEATURE_ABLATION.md'),Path('benchmarks/composition_features_2026-11-08.md')]
 identities=[]
 for f in files:
  if not f.is_file():continue
  dst=source/f;dst.parent.mkdir(parents=True,exist_ok=True);copy_file(f,dst);identities.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 env={'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','timm','transformers','huggingface_hub','bitsandbytes','onnxruntime','peft']},'source_parent':subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True),'source_files':identities,'packet_membership':'completed first-gate and auxiliary packets, source snapshots and completed composition logs; HF transfer metadata and mutable goal progress excluded'}
 (a.output/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(json.dumps({'packet':str(a.output),'source_files':len(identities)}))
if __name__=='__main__':main()
