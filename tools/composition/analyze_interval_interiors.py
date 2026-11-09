#!/usr/bin/env python3
"""Separate unique interval raw-score evidence from mode normalization artifacts."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha
from distill_attention_linear import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','anchors','output']:p.add_argument(name,type=Path)
    a=p.parse_args();rows=[json.loads(line) for line in (a.study/'predictions.jsonl').read_text().splitlines()];anchors={(r['id'],r['query']):r for r in json.loads(a.anchors.read_text())};groups=[]
    for seed in [31,137,911]:
        for policy in ['parent','refined']:
            for secondary in [False,True]:
                selected=[r for r in rows if r['seed']==seed and r['policy']==policy and (r['ratio_index']>=9)==secondary and r['query']%2==1 and r['valid'] and r['offered']];dips=[min(r['raw_scores'][0],r['raw_scores'][-1])-min(r['raw_scores'][1:-1]) for r in selected];groups.append(dict(seed=seed,policy=policy,secondary=secondary,unique_intervals_with_valid_maximal_query=len(selected),raw_interior_dips=sum(v>0 for v in dips),worst_raw_drop=max(dips) if dips else None))
    offered=[r for r in rows if r['valid'] and r['offered']];worst=max(offered,key=lambda r:r['worst_interior_drop_from_worse_endpoint']);anchor=anchors[worst['id'],worst['query']];write(a.output,dict(inputs={str(path):sha(path) for path in [a.study/'predictions.jsonl',a.anchors]},groups=groups,worst_standardized_witness=dict(record=worst,original_anchor=anchor,raw_drop=min(worst['raw_scores'][0],worst['raw_scores'][-1])-min(worst['raw_scores'][1:-1])),limitation='Original Minimal anchors cover only fixed maximum extent; interior extents extrapolate outside that normalization domain. Raw-score dips are finite teacher diagnostics, not human quality defects or excellent interval admission. Paired modes are not independent samples.',api_calls=0,test_inference=False,model_promoted=False));print(json.dumps(groups))


if __name__=='__main__':main()
