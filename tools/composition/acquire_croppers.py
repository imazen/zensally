#!/usr/bin/env python3
"""Pinned bounded acquisition of closer local crop teachers; preserve all identities."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi,snapshot_download
from acquire_teachers import fetch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
 if shutil.disk_usage(a.output).free<10*1024**3:raise ValueError('need ten GiB disk headroom')
 for repo,rev,name in [('appwrite/focalnet','f3c55aeb5827f302d1bbe9f0f41da38139f3dc36','focalnet-source'),('JianzeLi-114/FrameLite','e0f28dd2b6e137264a0a5a9996c1cca17c330657','framelite-source')]:
  tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1',8*1024**2));chosen=[r for r in tree['tree'] if r['type']=='blob' and (r['path'].endswith('.py') or Path(r['path']).name.lower() in ['readme.md','license','notice','requirements.txt','pyproject.toml'])]
  if tree.get('truncated') or len(chosen)>500 or sum(r.get('size',0) for r in chosen)>32*1024**2:raise ValueError('source bound exceeded')
  def obtain(r):
   path=a.output/name/r['path']
   if not path.resolve().is_relative_to((a.output/name).resolve()):raise ValueError('source path escaped')
   url=f'https://raw.githubusercontent.com/{repo}/{rev}/{r["path"]}';b=fetch(url,2*1024**2);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);return {'path':str(path.relative_to(a.output)),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
  with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(obtain,chosen))
  records.append({'repo':repo,'revision':rev,'files':files});print(json.dumps({'source':repo,'files':len(files)}),flush=True)
 for repo,rev,name,selected in [('appwrite/focalnet','f114d602b055aa1955f23a591a161336517789c4','focalnet',['focalnet-human.onnx','focalnet-human.onnx.json','focalnet-human.pt','README.md']),('zlab-sjtu/FrameLite','5c3c9aa1387ed2492f7e7816934167a720df8842','framelite',['framelite.safetensors','inference_config.json','LICENSE','NOTICE','README.md']),('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17','framelite/base',None)]:
  info=HfApi().model_info(repo,revision=rev,files_metadata=True);files=[f for f in info.siblings if f.rfilename in selected] if selected else [f for f in info.siblings if f.rfilename!='.gitattributes'];total=sum(f.size for f in files)
  if total>3*1024**3:raise ValueError('single model exceeds three GiB')
  dst=a.output/name;snapshot_download(repo,revision=rev,allow_patterns=[f.rfilename for f in files],local_dir=dst,max_workers=2);ids=[]
  for f in files:
   path=dst/f.rfilename
   if path.stat().st_size!=f.size:raise ValueError('asset size differs')
   with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   ids.append({'path':str(path.relative_to(a.output)),'bytes':f.size,'sha256':digest,'url':f'https://huggingface.co/{repo}/resolve/{rev}/{f.rfilename}'})
  records.append({'repo':repo,'revision':rev,'files':ids});(a.output/'asset-identities.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'model':repo,'files':len(files),'bytes':total}),flush=True)
if __name__=='__main__':main()
