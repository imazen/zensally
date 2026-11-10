#!/usr/bin/env python3
"""Bounded pinned local critic download; no inference service or paid API."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--name',default='photoframer');p.add_argument('--catalog',type=Path);a=p.parse_args()
if Path(a.name).name!=a.name:raise ValueError('model name must be a single directory name')
catalog=json.loads((a.catalog or a.root/'photoframer-catalog.json').read_text());output=a.root/a.name;output.mkdir(exist_ok=False)
selected=[x['name'] for x in catalog['files'] if x['name'] not in ['.gitattributes','training_args.bin']];total=sum(x['bytes'] for x in catalog['files'] if x['name'] in selected)
if total>20*1024**3 or shutil.disk_usage(a.root).free<total+8*1024**3:raise ValueError('critic exceeds 20-GiB/download or disk-headroom budget')
(output/'progress.json').write_text(json.dumps({'pid':os.getpid(),'stage':'downloading pinned critic','bytes_required':total,'revision':catalog['revision'],'checked_unix':time.time()}))
snapshot_download(repo_id=catalog['repo'],revision=catalog['revision'],allow_patterns=selected,local_dir=output,max_workers=2)
records=[]
for row in catalog['files']:
 if row['name'] not in selected:continue
 path=output/row['name']
 if path.stat().st_size!=row['bytes']:raise ValueError('download size differs')
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 records.append({**row,'sha256':digest,'url':f"https://huggingface.co/{catalog['repo']}/resolve/{catalog['revision']}/{row['name']}"})
(output/'asset-identities.json').write_text(json.dumps({'repo':catalog['repo'],'revision':catalog['revision'],'files':records,'bytes':total},indent=2)+'\n')
(output/'progress.json').write_text(json.dumps({'pid':None,'stage':'complete','bytes':total,'files':len(records),'revision':catalog['revision'],'checked_unix':time.time()}));print(json.dumps({'bytes':total,'files':len(records),'revision':catalog['revision']}))
