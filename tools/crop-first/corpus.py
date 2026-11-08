#!/usr/bin/env python3
"""Freeze current export/family identities; historical evaluation stays excluded."""
import argparse,csv,hashlib,io,json
from collections import Counter,defaultdict
from pathlib import Path
from urllib.request import Request,urlopen

CORPUS='32ba0ebea7b3cccbd6e141a397d90423140a7fc6'
OLD='b1faae550eba4d29efee1de1a32954ef11a0e1ed'
EXPORT={'train':'71d277215eaac178fd13c0bd6f6bf9fce333d9a6','validate':'838eb81fb50e52caa007143646c2eb8f2608968b','test':'b388ff04b21857a73b3ba0f5ef5ff0e0b4ba5d3b'}
RATIOS=['1:1','4:3','3:4','16:9','9:16','2:1','1:2','3:1','1:3','6:1','1:6']
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def rows(b):return list(csv.DictReader(io.StringIO(b.decode()),delimiter='\t'))
def fetch(root,name,url,links):
    p=root/name
    if p.exists():raise ValueError('freeze requires fresh manifest files')
    with urlopen(Request(url,headers={'User-Agent':'zensally-crop-research'}),timeout=90) as r:b=r.read()
    p.write_bytes(b);links.append({'text':name,'href':url,'sha256':sha(b)})
    return rows(b)

def reconcile(exports,families,historical):
    """The exporter specifies '1'+old ID. Hash equality is separate evidence."""
    fm={r['id']:r for r in families};blocked=set()
    for r in historical:
        if r['split']!='train':
            f=fm.get(r['id'])
            if f:blocked.add(f['family'] or f['id'])
    result=[];seen=set();family_roles=defaultdict(set)
    for role,rs in exports.items():
        for r in rs:
            sid=r['id']
            if len(sid)!=5 or sid[0]!='1' or sid in seen:raise ValueError('invalid/duplicate export identity')
            seen.add(sid);old=sid[1:];f=fm[old];group=f['family'] or old
            if r['bucket']!=role or f['split']!=role:raise ValueError('export/family role mismatch')
            provenance=json.loads(r['provenance_json'])
            if provenance['id']!=sid or provenance['split']!=role or provenance['original_sha256']!=r['original_sha256']:raise ValueError('provenance identity mismatch')
            family_roles[group].add(role)
            result.append({**r,'old_source_id':old,'family_id':group,'historical_eval_family':group in blocked,'category':provenance['category'],'provenance':provenance})
    if any(len(v)!=1 for v in family_roles.values()):raise ValueError('family crosses export roles')
    return result

def select(records,counts):
    chosen=[];used=set()
    for use,n in counts.items():
        role='train' if use in ['fit','dev'] else 'validate' if use=='eval' else 'test'
        pool=[r for r in records if r['bucket']==role and (role!='train' or not r['historical_eval_family']) and int(r['width'])*int(r['height'])<=32_000_000 and min(int(r['width']),int(r['height']))>=128]
        groups=defaultdict(list)
        for r in pool:groups[r['category']].append(r)
        for v in groups.values():v.sort(key=lambda r:sha(('zensally-fit-v1|'+r['id']).encode()))
        count=0
        while count<n:
            progress=False
            for category in sorted(groups):
                candidates=groups[category]
                while candidates and candidates[0]['family_id'] in used:candidates.pop(0)
                if candidates and count<n:
                    r=candidates.pop(0);chosen.append({**r,'use':use});used.add(r['family_id']);count+=1;progress=True
            if not progress:raise ValueError('insufficient distinct eligible source families')
    return chosen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--fit',type=int,default=64);p.add_argument('--dev',type=int,default=16);p.add_argument('--eval',type=int,default=16);p.add_argument('--test',type=int,default=8);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);links=[]
    family=fetch(a.output,'family-v2.tsv',f'https://raw.githubusercontent.com/imazen/imazen-26/{CORPUS}/manifests/split_map_family_v2.tsv',links)
    historical=[]
    for role in ['train','validate','test']:
        historical+=fetch(a.output,f'historical-{role}.tsv',f'https://raw.githubusercontent.com/imazen/imazen-26/{OLD}/manifests/{role}.tsv',links)
    exports={role:fetch(a.output,f'jxl-{role}.tsv',f'https://raw.githubusercontent.com/imazen/imazen-26-jxl-srgb/{rev}/manifest.tsv',links) for role,rev in EXPORT.items()}
    records=reconcile(exports,family,historical);selected=select(records,{k:getattr(a,k) for k in ['fit','dev','eval','test']})
    save(a.output/'sources.json',selected)
    save(a.output/'protocol.json',{'schema':1,'canonical_roles':dict(Counter(r['bucket'] for r in records)),'selected_uses':dict(Counter(r['use'] for r in selected)),'ratios':RATIOS,'essential_ratio_range':['1:3','3:1'],'secondary_ratios':['1:6','6:1'],'canonical_holdouts_for_evaluation_only':True,'fit_selection_uses_train_only':True,'human_rated':False,'admission':'TRAIN-only noisy teacher pilot authorized 2026-11-08; rendered labels still require validation','source_id_rule':'scripts/render_v4_list.py new_id: 1 + four-digit source ID','family_map_commit':CORPUS,'historical_manifest_commit':OLD,'export_commits':EXPORT,'links':links})
    (a.output/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in links))
    print(json.dumps({'selected_uses':dict(Counter(r['use'] for r in selected)),'families':len({r['family_id'] for r in selected}),'historical_eval_families_excluded_from_train':len({r['family_id'] for r in records if r['bucket']=='train' and r['historical_eval_family']})}))
if __name__=='__main__':main()
