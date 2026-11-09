#!/usr/bin/env python3
"""Fixed-bank local face admissibility, separate from weak-label evaluation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch
from artifact_identity import sha
from subject_origin_projection import padded_heads
from subject_policy import face_relation


DETECTORS = ['native-yunet', 'native-yunet-tiled-interior',
             'float16-blaze_face_full_range_sparse']


def allowed(box, heads, require_face):
    relations = [face_relation(box, head) for head in heads]
    return 'cut' not in relations and (not require_face or 'contained' in relations)


def summarize(rows):
    families = defaultdict(list)
    for row in rows:
        families[row['family_id']].append(row)
    result = dict(requests=len(rows), families=len(families))
    for key in ['offered', 'cut', 'match', 'no_detections', 'changed', 'excludes_all_detections']:
        result[key] = sum(sum(r[key] for r in group) / len(group)
                          for group in families.values()) / len(families)
    offered = [r for r in rows if r['offered']]
    result.update(offered_requests=len(offered), contain_requests=len(rows)-len(offered),
                  weak_cut_requests=sum(r['cut'] for r in rows),
                  weak_match_requests=sum(r['match'] for r in rows),
                  weak_cut_fraction_among_offered=(sum(r['cut'] for r in offered)/len(offered)
                                                   if offered else None))
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'control', 'detections', 'output']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    if sha(a.cache) != 'a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3':
        raise ValueError('frozen cache differs')
    if sha(a.control/'SHA256SUMS.json') != '941d64dca4c3f0cf5477fdd78ac332dee1a080818c825e44b7c8f17043bb6fc4':
        raise ValueError('frozen rehearsal manifest differs')
    members = json.loads((a.control/'SHA256SUMS.json').read_text())
    c = torch.load(a.cache, map_location='cpu', weights_only=False)
    dev = [i for i, s in enumerate(c['sources']) if s['role']=='dev']
    if len(dev)!=24 or any(s['role'] not in ['fit', 'dev'] for s in c['sources']):
        raise ValueError('registered roles differ; TEST remains closed')
    local = {(r['id'], r['model']): r['boxes']
             for r in json.loads(a.detections.read_text())}
    scores = {'frozen-original': c['initial_scores'][dev]}
    weights = {}
    for arm in ['replay-only', 'replay-contrastive']:
        for seed in [31, 137, 911]:
            name = f'{arm}-seed{seed}'
            relative = f'relationship-rehearsal/{name}/selected.pt'
            path = a.control/relative
            if sha(path)!=members[relative]:
                raise ValueError('selected checkpoint differs')
            scores[name] = torch.load(path, map_location='cpu', weights_only=False)['scores']
            weights[name] = sha(path)
    if any(v.shape!=(24, 22, 128) or not torch.isfinite(v).all() for v in scores.values()):
        raise ValueError('registered finite DEV scores required')
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(cache_sha256=sha(a.cache), control_manifest_sha256=sha(a.control/'SHA256SUMS.json'),
                    detections_sha256=sha(a.detections), selected_weights_sha256=weights,
                    detectors=DETECTORS, padding_percent=[0, 10], require_face=[False, True],
                    candidate_geometry='existing normalized candidates rounded to native integer edges, as in actual-pixel viewer',
                    detector_geometry='floor left/top, ceil right/bottom; existing padded_heads rule',
                    fallback='contain when no eligible admissible candidate; no reranking, fitting or score changes',
                    no_detection='unguarded when require_face=false; contain when require_face=true',
                    evaluation='separate existing weak nominated-head negative and accepted-reference masks; unknown remains unknown',
                    cohort='DEV essential accepted-reference requests only; family-balanced; all seeds reported without winner selection',
                    api_calls=0, test_inference=False, model_promoted=False, face_coverage_certified=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for j, i in enumerate(dev):
        source = c['sources'][i]
        w, h = source['native_size']
        scale = torch.tensor([w, h, w, h])
        boxes = (c['query_boxes'][i]*scale).round().int().tolist()
        masks = {}
        counts = {}
        for detector in DETECTORS:
            heads = []
            for b in local[(source['id'], detector)]:
                head = [max(0, math.floor(b[0]*w)), max(0, math.floor(b[1]*h)),
                        min(w, math.ceil(b[2]*w)), min(h, math.ceil(b[3]*h))]
                if head[0]>=head[2] or head[1]>=head[3]:
                    raise ValueError('invalid detector head')
                heads.append(head)
            for padding in [0, 10]:
                padded = padded_heads(w, h, heads, padding)
                for require in [False, True]:
                    key = f'{detector}/pad{padding}/require{int(require)}'
                    masks[key] = torch.tensor([[allowed(b, padded, require) for b in query]
                                               for query in boxes]) & c['eligible'][i]
                    counts[key] = (heads, padded)
        for model, values in scores.items():
            for q in range(18):
                if not bool(c['reference_valid'][i, q].any()):
                    continue
                raw = int(values[j, q].masked_fill(~c['eligible'][i, q], -torch.inf).argmax())
                for policy in ['raw', *masks]:
                    mask = c['eligible'][i, q] if policy=='raw' else masks[policy][q]
                    offered = bool(mask.any())
                    selected = int(values[j, q].masked_fill(~mask, -torch.inf).argmax()) if offered else None
                    heads, padded = counts[policy] if policy!='raw' else ([], [])
                    if offered and policy!='raw' and not allowed(boxes[q][selected], padded, policy.endswith('require1')):
                        raise ValueError('selected crop violates supplied detector geometry')
                    rows.append(dict(id=source['id'], family_id=source['family_id'], query=q,
                                     ratio=c['ratios'][q//2], mode='minimal' if q%2==0 else 'maximal',
                                     model=model, policy=policy, selected=selected, raw_selected=raw,
                                     offered=offered, cut=offered and bool(c['negative'][i, q, selected]),
                                     match=offered and bool(c['positive'][i, q, selected]),
                                     changed=selected!=raw, no_detections=policy!='raw' and not heads,
                                     excludes_all_detections=offered and bool(heads) and all(
                                         face_relation(boxes[q][selected], head)=='excluded' for head in heads),
                                     admissible_candidates=int(mask.sum()),
                                     native_box=boxes[q][selected] if offered else None))
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['policy']].append(row)
    summaries = [dict(model=model, policy=policy, **summarize(group))
                 for (model, policy), group in groups.items()]
    if any(r['requests']!=139 or r['families']!=17 for r in summaries):
        raise ValueError('frozen essential DEV cohort differs')
    (a.output/'predictions.json').write_text(json.dumps(rows, separators=(',', ':'))+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(summary=summaries, api_calls=0,
        face_coverage_certified=False, model_promoted=False, test_inference=False), indent=2)+'\n')
    print(json.dumps([r for r in summaries if r['model']=='replay-contrastive-seed31']))


if __name__=='__main__':
    main()
