#!/usr/bin/env python3
"""Pin public teacher source snapshots and weights; bounded, no paid endpoints."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from urllib.request import Request,urlopen


def fetch(url,bound):
 with urlopen(Request(url,headers={'User-Agent':'zensally-research/1'}),timeout=90) as r:
  body=r.read(bound+1)
 if len(body)>bound:raise ValueError('download exceeds declared bound')
 return body

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
 for repo in ['fkryan/gazelle','facebookresearch/dinov2','matthias-k/DeepGaze']:
  info=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits?per_page=1',2*1024*1024))[0];commit=info['sha'];name=repo.split('/')[-1];dst=a.output/name;dst.mkdir()
  tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',8*1024*1024))
  if tree.get('truncated'):raise ValueError('source tree truncated')
  chosen=[x for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith('.py') or Path(x['path']).name.lower() in ['readme.md','license','setup.cfg','pyproject.toml','requirements.txt'])]
  if len(chosen)>500 or sum(x.get('size',0) for x in chosen)>32*1024*1024:raise ValueError('selected source exceeds bound')
  def obtain(item):
   rel=Path(item['path']);target=dst/rel
   if not target.resolve().is_relative_to(dst.resolve()):raise ValueError('unsafe source path')
   url=f'https://raw.githubusercontent.com/{repo}/{commit}/{item["path"]}';data=fetch(url,2*1024*1024);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   return {'path':item['path'],'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(obtain,chosen))
  record={'repo':repo,'commit':commit,'source_files':files};records.append(record);print(json.dumps({'repo':repo,'commit':commit,'files':len(files)}),flush=True)
  (a.output/'source-identities.json').write_text(json.dumps(records,indent=2)+'\n')
 for name,url in [('gazelle-vitb-inout.pt','https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt'),('deepgaze3.pth','https://github.com/matthias-k/DeepGaze/releases/download/v1.1.0/deepgaze3.pth'),('centerbias_mit1003.npy','https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy')]:
  data=fetch(url,512*1024*1024);(a.output/name).write_bytes(data);records.append({'name':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});(a.output/'asset-identities.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records[-1]),flush=True)
if __name__=='__main__':main()
