#!/usr/bin/env python3
"""Bounded aspect-following diagnostic; never repair or admit model answers."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fit_heads import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path)
    parser.add_argument('first_gate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(args.models / 'framelite-source'))
    from framelite import FrameLite
    from framelite.geometry import InvalidCropError, parse_crop
    from framelite.model import _generation_prompt
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.manual_seed(31)
    model = FrameLite.from_pretrained(args.models / 'framelite', device='cuda:0')
    old = args.models / 'framelite-probe'
    old_protocol = json.loads((old / 'protocol.json').read_text())
    ratios = [(3, 1), (1, 3), (6, 1), (1, 6)]
    template = (
        '{base} The original image is {w} pixels wide and {h} pixels high. '
        'The crop must have width:height exactly {rw}:{rh} in original-image pixels. '
        'Return normalized coordinates as {{"crop":[left,top,right,bottom]}}. '
        'Never cut off any face. If no such crop is possible, return {{"crop":null}}.'
    )
    protocol = dict(old_protocol, parent_protocol_sha256=sha(old / 'protocol.json'),
                    prompt_template=template, ratios=ratios,
                    aspect_diagnostic='relative source-pixel ratio error <=1 percent, no repair',
                    source_preprocess='exact retained owned RGB inputs from the free-ratio probe',
                    face_coverage_certified=False, new_labels_admitted=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for source_id in old_protocol['sources']:
        parent = json.loads((old / (source_id + '.json')).read_text())
        raw = old / (source_id + '.rgb8')
        if sha(raw) != parent['owned_input_sha256']:
            raise ValueError('retained inference pixels changed')
        width, height = parent['source_size']
        dw, dh = parent['input_size']
        image = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(dh, dw, 3).copy()
        heads = [p['head'] for p in json.loads((args.first_gate / 'gaze-all' / (source_id + '.json')).read_text())['result']['people']]
        for rw, rh in ratios:
            prompt = template.format(base=model.config['prompt'], w=width, h=height, rw=rw, rh=rh)
            encoded = model.processor(text=[_generation_prompt(model.processor, prompt)],
                images=[image], do_resize=False, padding=True, truncation=False, return_tensors='pt').to('cuda')
            if encoded['image_grid_thw'].cpu().tolist() != [[1, dh // 16, dw // 16]]:
                raise ValueError('processor changed retained input dimensions')
            prompt_width = encoded.input_ids.shape[1]
            with torch.inference_mode():
                generated = model.model.generate(**encoded, **model.config['generation'],
                    eos_token_id=model.processor.tokenizer.eos_token_id,
                    pad_token_id=model.processor.tokenizer.pad_token_id)
            tokens = generated[0, prompt_width:]
            answer = model.processor.batch_decode([tokens], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            box, error, abstain = None, None, False
            try:
                abstain = answer == '{"crop":null}'
                if not abstain:
                    box = list(parse_crop(answer))
            except InvalidCropError as exception:
                error = str(exception)
            ratio_error, cuts = None, None
            if box:
                ratio = (box[2] - box[0]) * width / ((box[3] - box[1]) * height)
                ratio_error = abs(ratio / (rw / rh) - 1)
                cuts = sum(not (box[0] <= h[0] and box[1] <= h[1] and box[2] >= h[2] and box[3] >= h[3]) for h in heads)
            row = dict(id=source_id, ratio=f'{rw}:{rh}', prompt=prompt, raw_text=answer,
                completion_token_ids=tokens.cpu().tolist(), bbox_normalized=box,
                explicit_abstention=abstain, error=error, aspect_relative_error=ratio_error,
                aspect_within_one_percent=ratio_error is not None and ratio_error <= .01,
                known_heads=len(heads), cut_known_heads=cuts,
                rgb_sha256=parent['rgb_sha256'], owned_input_sha256=parent['owned_input_sha256'],
                source_size=parent['source_size'], input_size=parent['input_size'],
                eos_generated=bool((tokens == model.processor.tokenizer.eos_token_id).any()),
                face_coverage_certified=False, deployable=False)
            with (args.output / f'{source_id}-{rw}x{rh}.json').open('x') as stream:
                json.dump(row, stream, indent=2)
                stream.write('\n')
            rows.append(row)
            (args.output / 'progress.json').write_text(json.dumps(dict(pid=os.getpid(),
                completed=len(rows), required=32, checked_unix=time.time())))
            print(json.dumps({k: row[k] for k in ['id', 'ratio', 'bbox_normalized',
                'aspect_relative_error', 'cut_known_heads', 'explicit_abstention', 'error']}), flush=True)
    summary = dict(completed=len(rows), valid=sum(r['bbox_normalized'] is not None for r in rows),
        explicit_abstentions=sum(r['explicit_abstention'] for r in rows),
        aspect_within_one_percent=sum(r['aspect_within_one_percent'] for r in rows),
        outputs_cutting_known_heads=sum(bool(r['cut_known_heads']) for r in rows),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved(), new_labels_admitted=False,
        face_coverage_certified=False, independent_quality_claim=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'progress.json').write_text(json.dumps(dict(pid=None, stage='complete',
        completed=len(rows), required=32, checked_unix=time.time())))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
