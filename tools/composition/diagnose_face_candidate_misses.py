#!/usr/bin/env python3
"""Separate weak-head detector misses from fixed-bank coverage; no new labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import torch
from artifact_identity import sha
from subject_origin_projection import padded_heads
from subject_policy import face_relation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'crop-first'))
from subject_geometry import pixels


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'direct', 'audit', 'detections']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    if sha(a.cache)!='a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3' or sha(a.direct)!='2556ff1a29837918d3c383276fe64542157a8cfdb60133a90dc6d9ffee3f9472':
        raise ValueError('registered weak caches differ')
    c = torch.load(a.cache, map_location='cpu', weights_only=False)
    direct = torch.load(a.direct, map_location='cpu', weights_only=False)
    keys = lambda values: [(s['id'], s['role'], s['family_id']) for s in values]
    if keys(c['sources'])!=keys(direct['sources']):
        raise ValueError('weak heads cross source/role/family identities')
    protocol = json.loads((a.audit/'protocol.json').read_text())
    if sha(a.detections)!=protocol['detections_sha256'] or sha(a.cache)!=protocol['cache_sha256']:
        raise ValueError('audit inputs differ')
    local = {(r['id'], r['model']): r['boxes'] for r in json.loads(a.detections.read_text())}
    predictions = {(r['id'], r['query'], r['policy']): r
                   for r in json.loads((a.audit/'predictions.json').read_text())
                   if r['model']=='replay-contrastive-seed31'}
    policies = [name+'/pad10/require1' for name in ['native-yunet-tiled-interior',
                'union-tiled-landmarker', 'union-all-faces']]
    output = a.audit/'miss-diagnosis'
    output.mkdir(exist_ok=False)
    rows = []
    for i, source in enumerate(c['sources']):
        if source['role']!='dev':
            continue
        w, h = source['native_size']
        heads = [pixels(r, w, h, outward=True) for r in direct['head_rois'][i]]
        scale = torch.tensor([w, h, w, h])
        for q in range(18):
            if not bool(c['reference_valid'][i, q].any()):
                continue
            boxes = (c['query_boxes'][i, q]*scale).round().int().tolist()
            relations = [[face_relation(b, head) for head in heads] for b in boxes]
            cut = torch.tensor(['cut' in r for r in relations]) & c['eligible'][i, q]
            if not torch.equal(cut, c['negative'][i, q]):
                raise ValueError('native nominated heads do not reproduce frozen negatives')
            # Diagnostic only: these paid weak nominations are unavailable at runtime.
            oracle = c['eligible'][i, q] & ~cut & torch.tensor(['contained' in r for r in relations])
            for policy in policies:
                prediction = predictions[source['id'], q, policy]
                detector = policy.split('/')[0]
                detected = [b for member in protocol['detectors'][detector]
                            for b in local[source['id'], member]]
                detected = padded_heads(w, h, [[max(0, math.floor(b[0]*w)), max(0, math.floor(b[1]*h)),
                                               min(w, math.ceil(b[2]*w)), min(h, math.ceil(b[3]*h))]
                                              for b in detected], 10)
                selected = prediction['selected']
                misses = []
                if selected is not None:
                    if prediction['cut']!=bool(cut[selected]):
                        raise ValueError('prediction cut flag differs')
                    for head, relation in zip(heads, relations[selected]):
                        if relation!='cut':
                            continue
                        associated = [b for b in detected if overlap(head, b)>0]
                        states = {face_relation(boxes[selected], b) for b in associated}
                        if 'cut' in states:
                            raise ValueError('local supplied-head filter invariant failed')
                        category = ('no-overlapping-detection' if not associated else
                                    'overlap-with-retained-detection' if states=={'contained'} else
                                    'overlap-with-excluded-detection' if states=={'excluded'} else
                                    'overlap-with-retained-and-excluded-detections')
                        misses.append(dict(weak_head_native=head, category=category,
                            max_detection_coverage_of_weak_head=max((overlap(head, b)/((head[2]-head[0])*(head[3]-head[1])) for b in associated), default=0)))
                rows.append(dict(id=source['id'], family_id=source['family_id'], query=q,
                    policy=policy, offered=prediction['offered'], cut=prediction['cut'],
                    weak_safe_candidate_count=int(oracle.sum()),
                    weak_safe_positive_count=int((oracle & c['positive'][i, q]).sum()),
                    cut_heads=misses, source=source['source']))
    summary = []
    for policy in policies:
        selected = [r for r in rows if r['policy']==policy]
        cuts = [r for r in selected if r['cut']]
        summary.append(dict(policy=policy, requests=len(selected), cut_requests=len(cuts),
            cut_head_occurrences=sum(len(r['cut_heads']) for r in cuts),
            categories=dict(Counter(h['category'] for r in cuts for h in r['cut_heads'])),
            cut_requests_with_weak_safe_candidate=sum(r['weak_safe_candidate_count']>0 for r in cuts),
            cut_requests_with_accepted_positive_candidate=sum(r['weak_safe_positive_count']>0 for r in cuts),
            requests_without_weak_safe_candidate=sum(r['weak_safe_candidate_count']==0 for r in selected)))
    result = dict(summary=summary, direct_sha256=sha(a.direct),
        audit_predictions_sha256=sha(a.audit/'predictions.json'), native_weak_negative_replay_exact=True,
        explanation='Positive overlap is a geometric diagnostic, not a verified person association. Weak nominated heads are incomplete and unreviewed. Oracle candidate availability is diagnostic only and never a runtime guard or training admission.',
        api_calls=0, test_inference=False, model_promoted=False, face_coverage_certified=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'predictions.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
