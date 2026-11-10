#!/usr/bin/env python3
"""Bind cached phrase-grounded extents to unchanged synthetic descriptors."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch

from artifact_identity import sha

GROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('original_cache', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan_path = args.root/'source-plan.json'
    if sha(plan_path) != '6dc76f4d8730dcb91e17db659a5be330ebd18a88401b9caf2f594610590befa3':
        raise ValueError('registered source plan differs')
    plan = json.loads(plan_path.read_text())
    summary_path = args.root/'evaluation/primary56-summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['completed'] != 56 or summary['required'] != 56:
        raise ValueError('all registered calls must be retained before admission')
    calls = {r['id']:r for r in summary['calls']}
    planned = {r['id']:r for r in plan['sources']}
    old_metadata_path = args.original_cache.parent/'admission.json'
    old_metadata = json.loads(old_metadata_path.read_text())
    if sha(args.original_cache) != old_metadata['cache_sha256']:
        raise ValueError('original frozen descriptor identity differs')
    cache = torch.load(args.original_cache, map_location='cpu', weights_only=False)
    if cache['encoder_sha256'] != '910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71':
        raise ValueError('registered original encoder differs')
    if len(calls) != 56 or set(calls) != set(planned) or {s['id'] for s in cache['sources']} != set(planned):
        raise ValueError('source/query/descriptor sets differ')
    sources, targets, present, receipts, reasons = [], [], [], [], Counter()
    box_count = 0
    for source in cache['sources']:
        key = source['id']
        registered, call = planned[key], calls[key]
        if (source['use'] != registered['role'] or source['family_id'] != registered['family_id']
                or source['rgb_sha256'] != registered['rgb_sha256']):
            raise ValueError('descriptor source, pixel or family role differs')
        response_path = args.root/'evaluation/responses'/call['response']
        response = json.loads(response_path.read_text())
        contract = response['contract']
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        input_path = args.root/'evaluation/inputs'/(key+'.rgb')
        input_meta = json.loads(input_path.with_suffix('.json').read_text())
        if (digest != response_path.stem or digest != response['contract_sha256']
                or contract['source_id'] != key or contract['source_role'] != registered['role']
                or contract['source_plan_sha256'] != sha(plan_path)
                or contract['native_size'] != registered['native_size']
                or contract['task'] != GROUNDING or contract['prompt'] != GROUNDING+registered['query']
                or input_meta['contract']['source_rgb_sha256'] != source['rgb_sha256']
                or sha(input_path) != contract['owned_input_sha256']):
            raise ValueError('cached teacher contract does not bind to registered source')
        reason, boxes, entities = None, [], []
        if response['postprocess_error'] or response['token_cap_reached'] or not response['ended_with_eos']:
            reason = 'incomplete_generation_or_parser_error'
        else:
            parsed = response['parsed'][GROUNDING]
            if len(parsed['bboxes']) != len(parsed['labels']):
                raise ValueError('entity/box count differs')
            w, h = registered['native_size']
            for box, label in zip(parsed['bboxes'], parsed['labels']):
                valid = (len(box) == 4 and all(math.isfinite(v) for v in box)
                         and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)
                entities.append(dict(label=label, box=box, valid_geometry=valid))
                if not valid:
                    reason = 'invalid_native_geometry'
                else:
                    boxes.append([box[0]/w, box[1]/h, box[2]/w, box[3]/h])
                    box_count += 1
            if not boxes and reason is None:
                reason = 'no_entities_unknown'
        available = reason is None
        target = ([min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes)] if available else [0,0,1,1])
        targets.append(target)
        present.append(available)
        reasons[reason or 'present'] += 1
        sources.append(dict(source, primary_extent_training_eligible=source['use']=='fit' and available,
            primary_extent_present=available, target=target if available else None,
            target_origin='union of cached Florence phrase-grounded entities; weak, potentially incomplete or inflated',
            human_rated=False, face_coverage_certified=False))
        receipts.append(dict(id=key, response_sha256=sha(response_path), entities=entities,
                             query=registered['query'], unknown_reason=reason))
    output = args.root/'extent-cache'
    output.mkdir(exist_ok=False)
    result = dict(cache, targets=torch.tensor(targets, dtype=torch.float32), present=torch.tensor(present),
        sources=sources, teacher_identities=receipts, teacher_protocol_sha256=sha(summary_path),
        query_plan_sha256=sha(plan_path), original_descriptor_cache_sha256=sha(args.original_cache),
        schema=2, supervision='noisy_florence_phrase_grounded_primary_extents')
    torch.save(result, output/'features.pt')
    readback = torch.load(output/'features.pt', map_location='cpu', weights_only=False)
    if any(not torch.equal(readback[k], cache[k]) for k in ['visual', 'geometry']):
        raise ValueError('copied descriptor readback differs')
    metadata = {k:v for k,v in result.items() if k not in ['visual','geometry','targets','present']}
    metadata.update(cache_sha256=sha(output/'features.pt'), source_plan_sha256=sha(plan_path),
        fit_eligible=sum(s['primary_extent_training_eligible'] for s in sources),
        dev_eligible=sum(s['use']=='dev' and s['primary_extent_present'] for s in sources),
        status_counts=dict(reasons), valid_boxes=box_count, descriptors_bit_identical=True,
        model_promoted=False, paid_model_api_calls=0)
    with (output/'admission.json').open('x') as stream:
        stream.write(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['cache_sha256','fit_eligible','dev_eligible',
          'status_counts','valid_boxes','descriptors_bit_identical']}))


if __name__ == '__main__':
    main()
