#!/usr/bin/env python3
"""Condition frozen student proposals on explicit, unreviewed purpose regions."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch

from artifact_identity import sha
from constrained_bank import candidates


def contains(box, regions):
    l,t,r,b = box
    return all(l <= x and t <= y and r >= z and b >= v for x,y,z,v in regions)


def outward(box, width, height):
    return [math.floor(box[0]*width/100), math.floor(box[1]*height/100),
            math.ceil(box[2]*width/100), math.ceil(box[3]*height/100)]


def mode_indices(boxes, maximum_w, maximum_h, mode):
    selected = [i for i,(l,t,r,b) in enumerate(boxes) if
                ((r-l == maximum_w and b-t == maximum_h) if mode == 'minimal' else r-l < maximum_w)]
    fallback = mode == 'maximal' and bool(boxes) and not selected
    return (list(range(len(boxes))) if fallback else selected), fallback


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'annotations', 'export', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.annotations) != '226a230c8de06a9d6c83efd7a5e44a3c9f4905e12bb2c42e6d976f1a37c7b3a7':
        raise ValueError('registered unreviewed purpose annotations differ')
    annotation = json.loads(args.annotations.read_text())
    cache_path = args.study/'cache/cache.pt'
    if sha(cache_path) != '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031':
        raise ValueError('registered fit/development cache differs')
    manifest = {v['path']:v['sha256'] for v in json.loads((args.study/'SHA256SUMS.json').read_text())}
    prediction_path = args.study/'fits/trainable-image-seed911/best-predictions.json'
    if sha(prediction_path) != manifest['fits/trainable-image-seed911/best-predictions.json']:
        raise ValueError('frozen unconditioned predictions changed')
    predictions = {(v['id'],v['query']):v for v in json.loads(prediction_path.read_text())}
    qualification = json.loads((args.export/'qualification.json').read_text())
    if qualification['weights_sha256'] != 'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':
        raise ValueError('preselected student differs')
    for name in ['encoder', 'query']:
        if sha(args.export/(name+'.onnx')) != qualification['exports'][name]['sha256']:
            raise ValueError('qualified model graph changed')
    torch.set_num_threads(1)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    encoder, query = [session(args.export/(name+'.onnx')) for name in ['encoder', 'query']]
    args.output.mkdir(parents=True, exist_ok=False)
    totals, by_purpose, rows = Counter(), defaultdict(Counter), []
    for source in annotation['sources']:
        index = next(i for i,s in enumerate(cache['sources']) if s['id'] == source['id'])
        actual = cache['sources'][index]
        if (actual['role'] != 'dev' or source['role'] != 'dev' or
                actual['rgb_sha256'] != source['rgb_sha256'] or sha(Path(source['png'])) != source['png_sha256']):
            raise ValueError('development source identity changed')
        width, height = source['native_size']
        heads = [outward(b, width, height) for b in source['head_candidate_boxes_percent']]
        f8, f16 = encoder.run(None, dict(image=cache['image'][index:index+1].numpy()))
        image_name = source['id'].split(':',1)[1]+'.png'
        shutil.copy2(source['png'], args.output/image_name)
        tasks = []
        for purpose, nomination in source['requests'].items():
            focus = outward(nomination['required_box_percent'], width, height)
            required = heads+[focus]
            for ratio_index, ratio in enumerate(cache['ratios']):
                p,q = map(int, ratio.split(':'))
                divisor = math.gcd(p,q)
                p,q = p//divisor, q//divisor
                boxes, mw, mh = candidates(width, height, (p,q), required)
                # Independent closed-form feasibility from region-union extents.
                span_w = max(b[2] for b in required)-min(b[0] for b in required)
                span_h = max(b[3] for b in required)-min(b[1] for b in required)
                minimum_k = max((span_w+p-1)//p, (span_h+q-1)//q)
                maximum_k = min(width//p, height//q)
                feasible = minimum_k <= maximum_k
                if bool(boxes) != feasible:
                    raise ValueError('constrained bank disagrees with independent native feasibility')
                for l,t,r,b in boxes:
                    if not (0 <= l < r <= width and 0 <= t < b <= height and (r-l)*q == (b-t)*p
                            and contains((l,t,r,b), required)):
                        raise ValueError('candidate violates native geometry or a supplied region')
                padded = np.zeros((1,128,4), np.float32)
                if boxes:
                    padded[0,:len(boxes)] = np.array(boxes, np.float64)/[width,height,width,height]
                bank = cache['native_boxes'][index][ratio_index]
                safe_old = [b for b in bank if contains(b, required)]
                for mode_index, mode in enumerate(['minimal','maximal']):
                    qindex = ratio_index*2+mode_index
                    original_index = predictions[(source['id'], qindex)]['selected']
                    original_box = bank[original_index]
                    eligible, fallback = mode_indices(boxes, mw, mh, mode)
                    old_eligible, old_fallback = mode_indices(safe_old, mw, mh, mode)
                    chosen, score = None, None
                    if eligible:
                        raw = query.run(None, dict(f8=f8, f16=f16, boxes=padded,
                            source_aspect=np.array([width/height], np.float32),
                            target_aspect=np.array([p/q], np.float32), mode=np.array([mode_index], np.float32),
                            content=cache['content'][index:index+1].numpy()))[0][0]
                        if not np.isfinite(raw).all():
                            raise ValueError('nonfinite purpose-conditioned research score')
                        selected = max(eligible, key=lambda j:float(raw[j]))
                        chosen, score = boxes[selected], float(raw[selected])
                    original_violation = not contains(original_box, required)
                    violation = chosen is not None and not contains(chosen, required)
                    record = dict(purpose=purpose, nomination=nomination['description'], ratio=ratio, mode=mode,
                        original_box=original_box, conditional_box=chosen, score=score,
                        required_pixels=required, head_candidate_pixels=heads, focus_pixels=focus,
                        unconditioned_region_violation=original_violation, selected_region_violation=violation,
                        native_geometry_feasible=feasible, minimum_required_scale_units=minimum_k,
                        maximum_scale_units=maximum_k, old_filtered_bank_available=bool(old_eligible),
                        old_filtered_fallback=old_fallback, conditional_fallback=fallback,
                        origin_recovery=bool(eligible) and not old_eligible,
                        counterfactual_action='research_crop' if chosen else 'contain',
                        offered_action='contain', face_coverage_certified=False,
                        crop_area_fraction=None if chosen is None else (chosen[2]-chosen[0])*(chosen[3]-chosen[1])/(width*height))
                    tasks.append(record)
                    for counter in [totals, by_purpose[purpose]]:
                        counter['requests'] += 1
                        counter['research_crop' if chosen else 'contain'] += 1
                        counter['unconditioned_region_violations'] += int(original_violation)
                        counter['selected_region_violations'] += int(violation)
                        counter['origin_recoveries'] += int(record['origin_recovery'])
                        counter['maximal_fallbacks'] += int(fallback)
        rows.append(dict(id=source['id'], family_id=source['family_id'], role='dev', image=image_name,
            native_size=source['native_size'], source_url=source['source_url'], photographer=source['photographer'],
            detector_person_boxes=source['detector_person_boxes'], tasks=tasks))
        print(json.dumps(dict(completed_sources=len(rows), required_sources=6, summary=dict(totals))), flush=True)
    if totals['requests'] != 330 or totals['selected_region_violations']:
        raise ValueError('registered purpose comparison incomplete or violates supplied regions')
    result = dict(sources=rows, ratios=cache['ratios'], summary=dict(totals),
        by_purpose={k:dict(v) for k,v in by_purpose.items()}, annotations_sha256=sha(args.annotations),
        student_weights_sha256=qualification['weights_sha256'], model_has_purpose_input=False,
        training_updated=False, model_reselected=False, training_labels_admitted=False,
        human_artistic_quality_measured=False, face_coverage_certified=False,
        new_test_inference=False, diagnostic_maps_saved=False, paid_model_api_calls=0,
        detail_scale_limit='same registered minimum65% linear extent; no close-up quality claim')
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(summary=result['summary'], by_purpose=result['by_purpose'])))


if __name__ == '__main__':
    main()
