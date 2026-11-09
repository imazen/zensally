#!/usr/bin/env python3
"""Compare cached pose framing diagnostics for fixed choices; no aesthetic labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import torch
from artifact_identity import sha
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from composition_features import NAMES, candidate_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'audit', 'pose']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads((a.audit/'protocol.json').read_text())
    if sha(a.pose)!=protocol['pose_sha256'] or sha(a.cache)!=protocol['cache_sha256']:
        raise ValueError('fixed framing inputs differ')
    c = torch.load(a.cache, map_location='cpu', weights_only=False)
    lookup = {s['id']: i for i, s in enumerate(c['sources'])}
    poses = {r['id']: r['result'] for r in json.loads(a.pose.read_text())['rows']}
    policies = ['raw', 'union-all-faces/pad25/require1', 'union-all-faces-pose/pad25/require1']
    records = [r for r in json.loads((a.audit/'predictions.json').read_text())
               if r['policy'] in policies and r['model'].startswith('replay-contrastive-seed')]
    rows = []
    fields = ['head_joints_retained', 'shoulders_retained', 'hips_retained',
              'knees_retained', 'feet_retained', 'whole_pose_retained',
              'upper_arm_endpoint_cut', 'forearm_endpoint_cut', 'thigh_endpoint_cut',
              'shin_endpoint_cut', 'torso_endpoint_cut']
    for r in records:
        i = lookup[r['id']]
        if c['sources'][i]['role']!='dev':
            raise ValueError('DEV only')
        # Missing poses are unknown, and must not lower the mean cut diagnostic.
        if not poses[r['id']].get('landmarks'):
            continue
        if r['offered']:
            values = c['extras'][i, r['query'], r['selected'], :16].tolist()
        else:
            # Dummy scan data only satisfies the shared feature function's input
            # shape. None of its scan/gaze columns are read or saved here.
            scan = dict(paths=[[dict(x=.5, y=.5)]*6 for _ in range(4)])
            values = candidate_features([[0, 0, 1, 1]], poses[r['id']], dict(people=[]), scan)[0, :16].tolist()
        rows.append(dict(id=r['id'], family_id=r['family_id'], query=r['query'],
                         model=r['model'], policy=r['policy'], offered=r['offered'],
                         features={key:values[NAMES.index(key)] for key in fields}))
    summary = []
    for policy in policies:
        for offered_only in [False, True]:
            families = defaultdict(list)
            selected = [r for r in rows if r['policy']==policy and (r['offered'] or not offered_only)]
            for r in selected:
                families[r['family_id']].append(r)
            summary.append(dict(policy=policy, offered_only=offered_only,
                requests_per_seed=len(selected)//3, families=len(families),
                **{key:sum(sum(r['features'][key] for r in group)/len(group)
                           for group in families.values())/len(families) for key in fields}))
    keyed = {(r['model'], r['id'], r['query'], r['policy']):r for r in records}
    transitions = []
    base, pose = policies[1:]
    for seed in [31, 137, 911]:
        model = f'replay-contrastive-seed{seed}'
        paired = [(r, keyed[model, r['id'], r['query'], pose]) for r in records
                  if r['model']==model and r['policy']==base]
        transitions.append(dict(seed=seed, requests=len(paired),
            cut_repaired_while_offered=sum(x['cut'] and y['offered'] and not y['cut'] for x,y in paired),
            cut_removed_by_contain=sum(x['cut'] and not y['offered'] for x,y in paired),
            introduced_cut=sum(not x['cut'] and y['cut'] for x,y in paired),
            new_contain=sum(x['offered'] and not y['offered'] for x,y in paired),
            new_crops=sum(not x['offered'] and y['offered'] for x,y in paired)))
    result = dict(summary=summary, transitions=transitions,
        feature_names=fields, pose_sha256=sha(a.pose), cache_sha256=sha(a.cache),
        explanation='Family-balanced descriptive geometry, pooled equally sized seed panels. Exclude absent-pose sources. Offered uses unchanged cached normalized candidate features; contain uses full-source pose geometry. Confidence product/0.25 feature gate differs from strict min0.5 head-bound gate. Lower endpoint-cut features do not certify good body framing or face completeness.',
        api_calls=0, test_inference=False, model_promoted=False)
    output = a.audit/'framing-diagnostics.json'
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    with (a.audit/'framing-predictions.json').open('x') as stream:
        stream.write(json.dumps(rows, separators=(',', ':'))+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
