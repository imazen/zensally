#!/usr/bin/env python3
"""Seal all blinded observations before allowing candidate identity decoding."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from artifact_identity import sha


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def validate(row,case):
    if set(row)!={'case_id','preferred','A','B','evidence'} or row['case_id']!=case['case_id'] or row['preferred'] not in ['A','B','tie','neither'] or not isinstance(row['evidence'],str) or not row['evidence'].strip():raise ValueError('complete blinded preference and visible evidence required')
    for alias in ['A','B']:
        if set(row[alias])!={'minimal','maximal'}:raise ValueError('both matching-aspect endpoints required')
        for entry in row[alias].values():
            if set(entry)!={'quality','introduced_face_cut','person_fragment'} or type(entry['quality']) is not int or entry['quality'] not in range(4) or entry['introduced_face_cut'] not in ['yes','no','uncertain'] or entry['person_fragment'] not in ['yes','no','uncertain']:raise ValueError('complete bounded endpoint assessment required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['record','unblind']);p.add_argument('root',type=Path);a=p.parse_args();public=json.loads((a.root/'cases.json').read_text());cases={r['case_id']:r for r in public['cases']}
    if len(cases)!=32:raise ValueError('all32 source/ratio requests required')
    reviews=a.root/'reviews';reviews.mkdir(exist_ok=True)
    if a.action=='record':
        if (a.root/'unblinded-summary.json').exists():raise ValueError('do not relabel after unblinding')
        row=json.load(sys.stdin)
        if row.get('case_id') not in cases:raise ValueError('unknown case')
        case=cases[row['case_id']];validate(row,case)
        for pane in case['panes']:
            if sha(a.root/pane['file'])!=pane['sha256']:raise ValueError('review pixels changed')
        write(reviews/(row['case_id']+'.json'),dict(observation=row,case_pixels={p['file']:p['sha256'] for p in case['panes']},blind_key_sha256=public['blind_key_sha256'],reviewer='Codex coordinator visual inspection; not human or independent implementer',training_eligible=False));print(json.dumps(dict(recorded=row['case_id'],completed=len(list(reviews.glob('case*.json'))),required=32)));return
    paths=sorted(reviews.glob('case*.json'))
    if len(paths)!=32 or {p.stem for p in paths}!=set(cases):raise ValueError('seal all32 blinded observations before reading identity key')
    observations={}
    for path in paths:
        value=json.loads(path.read_text());case=cases[path.stem];validate(value['observation'],case)
        if value['blind_key_sha256']!=public['blind_key_sha256'] or value['case_pixels']!={p['file']:p['sha256'] for p in case['panes']}:raise ValueError('sealed review binding differs')
        for pane in case['panes']:
            if sha(a.root/pane['file'])!=pane['sha256']:raise ValueError('sealed pixels changed')
        observations[path.stem]=value['observation']
    keypath=a.root/'_BLIND_KEY.json'
    if sha(keypath)!=public['blind_key_sha256']:raise ValueError('identity key changed after display')
    key=json.loads(keypath.read_text());decoded=[];preferences=Counter();endpoints={};conflicts=[]
    for record in key['cases']:
        cid=record['case_id'];row=observations[cid];models=record['models'];winner=models.get(row['preferred'],row['preferred']);preferences[winner]+=1;panes={p['label']:p for p in cases[cid]['panes']}
        for mode in ['minimal','maximal']:
            if panes['A '+mode]['rgb_sha256']==panes['B '+mode]['rgb_sha256'] and row['A'][mode]!=row['B'][mode]:conflicts.append(dict(case_id=cid,mode=mode,kind='identical pixels received different ratings'))
        identical=all(panes['A '+m]['rgb_sha256']==panes['B '+m]['rgb_sha256'] for m in ['minimal','maximal'])
        if identical and row['preferred'] in ['A','B']:conflicts.append(dict(case_id=cid,kind='identical pairs received strict preference'))
        for alias,model in models.items():
            for mode in ['minimal','maximal']:endpoints.setdefault(model,[]).append(dict(case_id=cid,ratio=record['ratio'],mode=mode,**row[alias][mode]))
        decoded.append(dict(case_id=cid,id=record['id'],ratio=record['ratio'],models=models,winner=winner,observation=row,identical_pairs=identical))
    summary=dict(requests=32,sources=16,preferences=dict(preferences),models={model:dict(endpoints=len(rows),quality_counts=dict(Counter(str(r['quality']) for r in rows)),introduced_face_cuts=dict(Counter(r['introduced_face_cut'] for r in rows)),person_fragments=dict(Counter(r['person_fragment'] for r in rows))) for model,rows in endpoints.items()},identical_pixel_conflicts=conflicts,cases=decoded,scope='eval-only blinded assistant inspection; not independent human artistic labels, training supervision or interval certification',api_calls=0,test_inference=False,training_labels_admitted=False,model_promoted=False);write(a.root/'unblinded-summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))


if __name__=='__main__':main()
