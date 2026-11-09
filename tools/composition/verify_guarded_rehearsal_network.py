#!/usr/bin/env python3
"""Replay guarded selected-student decisions through the full ORT encoder/query."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from artifact_identity import sha
from export_unsplash_ranker import session
from audit_rehearsal_face_candidates import allowed
from subject_origin_projection import padded_heads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'audit', 'serving', 'detections']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    if sha(a.cache)!='bfd84ff0ea48a7c3a2d2d32082f3fd8a9bd4a6c59b576899a80c4a7812f873c9':
        raise ValueError('frozen image cache differs')
    data = torch.load(a.cache, map_location='cpu', weights_only=False)
    protocol = json.loads((a.audit/'protocol.json').read_text())
    if sha(a.detections)!=protocol['detections_sha256']:
        raise ValueError('frozen local detections differ')
    qualification = json.loads((a.serving/'qualification.json').read_text())
    student = protocol['selected_weights_sha256']['replay-contrastive-seed31']
    if student!='4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b' or student not in qualification['inputs'].values():
        raise ValueError('selected student and serving graph lineage differ')
    for name in ['encoder', 'candidate-query']:
        if sha(a.serving/(name+'.onnx'))!=qualification['exports'][name]['sha256']:
            raise ValueError('serving graph differs')
    policies = ['raw', 'union-all-faces/pad25/require1', 'union-all-faces-pose/pad25/require1']
    expected = {(r['id'], r['query'], r['policy']):r
                for r in json.loads((a.audit/'predictions.json').read_text())
                if r['model']=='replay-contrastive-seed31' and r['policy'] in policies}
    source_metadata = {s['id']:s for s in json.loads((a.audit/'sources.json').read_text())}
    pose = json.loads((a.audit/'pose-heads.json').read_text())
    if pose['pose_sha256']!=protocol['pose_sha256']:
        raise ValueError('pose head identity differs')
    local = {(r['id'], r['model']):r['boxes'] for r in json.loads(a.detections.read_text())}
    encoder, query = session(a.serving/'encoder.onnx'), session(a.serving/'candidate-query.onnx')
    rows = []
    for i, source in enumerate(data['sources']):
        if source['role']!='dev':
            continue
        if source!=source_metadata[source['id']]:
            raise ValueError('source identity, role or family differs')
        tasks = sorted({q for sid,q,policy in expected if sid==source['id']})
        if not tasks:
            continue
        f8, f16 = encoder.run(None, dict(image=data['image'][i:i+1].numpy()))
        w, h = source['native_size']
        heads = {}
        for policy in policies[1:]:
            native = []
            for member in protocol['detectors'][policy.split('/')[0]]:
                if member=='pose-head-bounds':
                    native.extend(pose['source_boxes'][source['id']])
                else:
                    for b in local[source['id'], member]:
                        native.append([max(0, math.floor(b[0]*w)), max(0, math.floor(b[1]*h)),
                                       min(w, math.ceil(b[2]*w)), min(h, math.ceil(b[3]*h))])
            heads[policy] = padded_heads(w, h, native, 25)
        for q in tasks:
            boxes = data['query_boxes'][i:i+1, q].numpy()
            p, r = map(int, data['ratios'][q//2].split(':'))
            scores = query.run(None, dict(f8=f8, f16=f16, boxes=boxes,
                source_aspect=np.array([w/h], np.float32), target_aspect=np.array([p/r], np.float32),
                mode=np.array([q%2], np.float32), content=data['content'][i:i+1].numpy()))[0][0]
            if not np.isfinite(scores).all():
                raise ValueError('nonfinite complete neural scores')
            native = (data['query_boxes'][i,q]*torch.tensor([w,h,w,h])).round().int().tolist()
            for policy in policies:
                reference = expected[source['id'], q, policy]
                mask = data['eligible'][i,q].numpy().copy()
                if policy!='raw':
                    mask &= np.array([allowed(b, heads[policy], True) for b in native])
                choice = int(np.where(mask, scores, -np.inf).argmax()) if mask.any() else None
                old = reference['selected']
                same_presence = (choice is None)==(old is None)
                edge = float(np.abs(boxes[0,choice]-boxes[0,old]).max()*128) if choice is not None and old is not None else 0.
                if int(mask.sum())!=reference['admissible_candidates']:
                    raise ValueError('guard reconstruction differs')
                rows.append(dict(id=source['id'], query=q, policy=policy, cached=old, ort=choice,
                    same_presence=same_presence, changed_index=old!=choice,
                    edge_movement_128_grid=edge, gate_passed=same_presence and edge<=3))
    if len(rows)!=417 or set(expected)!={(r['id'],r['query'],r['policy']) for r in rows}:
        raise ValueError('fixed full guarded evaluation cohort differs')
    result = dict(rows=rows, decisions=len(rows), changed_indices=sum(r['changed_index'] for r in rows),
        gate_violations=sum(not r['gate_passed'] for r in rows), max_edge_movement_128_grid=max(r['edge_movement_128_grid'] for r in rows),
        student_sha256=student, image_cache_sha256=sha(a.cache),
        serving_exports={name:qualification['exports'][name] for name in ['encoder','candidate-query']},
        audit_predictions_sha256=sha(a.audit/'predictions.json'),
        scope='full ORT gather encoder/query on cached256x256 owned images, three fixed masks; excludes image preprocessing and detector inference',
        api_calls=0, test_inference=False, model_promoted=False, face_coverage_certified=False)
    with (a.audit/'guarded-serving-check.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','serving_exports']}))
    if result['gate_violations']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
