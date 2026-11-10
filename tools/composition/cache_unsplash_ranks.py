#!/usr/bin/env python3
"""Cache local relative rankings on fixed fit/dev sources; TEST stays closed."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from artifact_identity import sha
from cached_focalnet import CachedContext, session
from constrained_bank import candidates
from focalnet_bank import RATIOS
from rank_targets import targets as relative_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['registry', 'models', 'split', 'resizer', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--test-plan', type=Path)
    args = parser.parse_args()
    if sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('registered provider-family roles differ')
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('owned neural-input resizer differs')
    full_path = args.models/'focalnet/focalnet-human.onnx'
    if sha(full_path) != '59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439':
        raise ValueError('registered teacher differs')
    if (sha(args.split/'analysis.onnx') != 'a5f322876a6d53b9221e1eca6d6a6b95e91b7b911e7003d660444492ca504768'
            or sha(args.split/'query.onnx') != '8d14feeb5218baea8d885bba17e5b13b21e5c4dc6b6942a4d5de63adc51dc7ad'):
        raise ValueError('qualified cached teacher split differs')
    torch.set_num_threads(2)
    registry = json.loads(args.registry.read_text())
    testing = args.test_plan is not None
    if testing:
        plan = json.loads(args.test_plan.read_text())
        if (plan['schema'] != 'unsplash-final-rank-comparison-v1' or plan['registry_sha256'] != sha(args.registry)
                or len(plan['models']) != 12 or plan['test_sources'] != 25):
            raise ValueError('frozen twelve-model final comparison required before TEST reference inference')
        for model in plan['models']:
            if sha(Path(model['path'])) != model['sha256']:
                raise ValueError('a model changed after final selection freeze')
        for source in ['training/local_rank_distillation.py', 'training/crop_regions.py',
                       'training/crop_ranker.py', 'training/model_v3ds.py', 'tools/composition/fit_unsplash_ranks.py']:
            expected = subprocess.check_output(['git', 'show', plan['build_commit']+':'+source])
            if Path(source).read_bytes() != expected:
                raise ValueError('student/evaluation source differs from frozen selection commit')
    wanted = ['test'] if testing else ['fit', 'dev']
    sources = [r for r in registry['sources'] if r['role'] in wanted]
    expected_count = 25 if testing else 119
    if Counter(r['role'] for r in sources) != (dict(test=25) if testing else dict(fit=94, dev=25)):
        raise ValueError('registered fit/dev cohort differs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(registry_sha256=sha(args.registry),
        teacher_sha256=sha(full_path), resizer_sha256=sha(args.resizer), sources=expected_count,
        fit_sources=0 if testing else 94, dev_sources=0 if testing else 25, test_sources=25 if testing else 0,
        test_plan_sha256=sha(args.test_plan) if testing else None, analysis_threads=4, query_threads=1,
        purpose='unspecified weak local relative rank', paid_model_api_calls=0), indent=2)+'\n')
    images, content = [], []
    for source in sources:
        path = Path(source['rgb'])
        if sha(path) != source['rgb_sha256']:
            raise ValueError('owned working pixels differ')
        width, height = source['working_size']
        tw, th = max(1, round(width*256/max(width, height))), max(1, round(height*256/max(width, height)))
        with path.open('rb') as stream:
            value = subprocess.check_output([str(args.resizer), str(width), str(height), str(width*3),
                ','.join(map(str, [0, 0, width, height, tw, th]))], stdin=stream)
        if len(value) != tw*th*3:
            raise ValueError('owned neural input length differs')
        image = np.zeros((256, 256, 3), np.uint8)
        ox, oy = (256-tw)//2, (256-th)//2
        image[oy:oy+th, ox:ox+tw] = np.frombuffer(value, np.uint8).reshape(th, tw, 3)
        tensor_path = args.output/(source['id'].split(':', 1)[1]+'.rgb')
        tensor_path.write_bytes(image.tobytes())
        images.append(image.transpose(2, 0, 1).astype(np.float32)/255)
        content.append([ox/256, oy/256, (ox+tw)/256, (oy+th)/256])
        source['neural_input_sha256'] = sha(tensor_path)
    image = np.stack(images)
    bounds = np.array(content, np.float32)
    witnesses = []
    for role in wanted:
        used = set()
        for target in [.5, 1., 2.]:
            choices = [i for i, s in enumerate(sources) if s['role'] == role and s['family_id'] not in used]
            index = min(choices, key=lambda i: (abs(math.log(sources[i]['native_size'][0]/sources[i]['native_size'][1]/target)), sources[i]['id']))
            witnesses.append(index)
            used.add(sources[index]['family_id'])
    context = CachedContext(args.split, threads=4, query_threads=1)
    full = session(full_path, threads=4)
    mean, std = np.array([.485, .456, .406], np.float32)[:, None, None], np.array([.229, .224, .225], np.float32)[:, None, None]
    boxes_all, scores_all, masks_all, valid_all, pair_all, pair_masks, targets_all, native_boxes = [], [], [], [], [], [], [], []
    checks, counts = [], Counter()
    order = witnesses+[i for i in range(len(sources)) if i not in witnesses]
    collected = {}
    for index in order:
        source = sources[index]
        normalized = np.zeros((1, 3, 256, 256), np.float32)
        x0, y0, x1, y1 = np.rint(bounds[index]*256).astype(int)
        normalized[0, :, y0:y1, x0:x1] = ((image[index]-mean)/std)[:, y0:y1, x0:x1]
        token = context.analyze(normalized, bounds[index:index+1], (source['id'], source['rgb_sha256']))
        boxes, scores, masks, valid, pairs, pairs_valid, targets, original_boxes = [], [], [], [], [], [], [], []
        width, height = source['native_size']
        for ratio, _ in RATIOS:
            p, q = map(int, ratio.split(':'))
            native, mw, mh = candidates(width, height, (p, q), [])
            for l, t, r, b in native:
                if not (0 <= l < r <= width and 0 <= t < b <= height and (r-l)*q == (b-t)*p):
                    raise ValueError('native integer candidate geometry differs')
            padded = np.zeros((1, 128, 4), np.float32)
            padded[0, :len(native)] = np.array(native, np.float64)/[width, height, width, height]
            value = context.score(padded, token)[0].copy()
            if index in witnesses:
                reference = full.run(['crop_scores'], dict(image=normalized, boxes=padded, content=bounds[index:index+1]))[0][0]
                exact = np.array_equal(value.view(np.uint32), reference.view(np.uint32))
                checks.append(dict(id=source['id'], ratio=ratio, score_bits_exact=exact,
                    max_absolute_error=float(np.abs(value-reference).max())))
                if not exact:
                    raise ValueError('new source cached/full score bits differ')
            boxes.append(padded[0])
            scores.append(value)
            original_boxes.append(native)
            for mode in ['minimal', 'maximal']:
                selected = [j for j, b in enumerate(native) if (b[2]-b[0] == mw if mode == 'minimal' else b[2]-b[0] < mw)]
                if not selected and mode == 'maximal':
                    selected = list(range(len(native)))
                mask, available, target, pair, pair_mask = relative_targets(value, selected)
                masks.append(mask); valid.append(available); targets.append(target); pairs.append(pair); pairs_valid.append(pair_mask)
                counts['queries'] += 1
                counts['ranking_queries'] += int(available)
                counts['pairs'] += int(pair_mask.sum())
        collected[index] = (boxes, scores, masks, valid, pairs, pairs_valid, targets, original_boxes)
        (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),
            stage='cached-full-scout' if len(collected) < 6 else 'weak-rank-cache',
            completed=len(collected), required=expected_count, checked_unix=time.time()))+'\n')
        print(json.dumps(dict(completed=len(collected), required=expected_count, id=source['id'], role=source['role'])), flush=True)
    for index in range(len(sources)):
        b, s, m, v, p, pm, t, nb = collected[index]
        boxes_all.append(b); scores_all.append(s); masks_all.append(m); valid_all.append(v)
        pair_all.append(p); pair_masks.append(pm); targets_all.append(t); native_boxes.append(nb)
    cache = dict(schema='unsplash-local-rank-cache-v1', sources=sources, image=torch.from_numpy(image),
        content=torch.from_numpy(bounds), boxes=torch.tensor(np.array(boxes_all)),
        teacher_scores=torch.tensor(np.array(scores_all)), eligible=torch.tensor(np.array(masks_all)),
        query_valid=torch.tensor(np.array(valid_all)), pairs=torch.tensor(np.array(pair_all)),
        pair_valid=torch.tensor(np.array(pair_masks)), targets=torch.tensor(np.array(targets_all)),
        native_boxes=native_boxes, ratios=[r for r, _ in RATIOS], registry_sha256=sha(args.registry),
        teacher_sha256=sha(full_path), purpose='unspecified weak local relative rank',
        training_label_origin='local FocalNet relative ranking, not human acceptability',
        canonical_holdouts_opened=False, new_test_composition_scores_opened=testing,
        face_coverage_certified=False, diagnostic_maps_saved=False)
    if testing:
        cache.update(test_plan_sha256=sha(args.test_plan), training_eligible=False)
    torch.save(cache, args.output/'cache.pt')
    (args.output/'cached-full-scout.json').write_text(json.dumps(checks, indent=2)+'\n')
    summary = dict(sources=expected_count, fit_sources=0 if testing else 94, dev_sources=0 if testing else 25, counts=dict(counts),
        cache_sha256=sha(args.output/'cache.pt'), cache_bytes=(args.output/'cache.pt').stat().st_size,
        full_cached_scout_checks=len(checks), full_cached_scout_all_bits_exact=all(c['score_bits_exact'] for c in checks),
        new_test_sources_opened=25 if testing else 0, teacher_sha256=cache['teacher_sha256'], purpose=cache['purpose'],
        human_rated=False, face_coverage_certified=False, paid_model_api_calls=0)
    if summary['cache_bytes'] > 256*1024**2:
        raise ValueError('256MiB input/target cache budget exhausted; preserve artifacts')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete',
        completed=expected_count, required=expected_count, checked_unix=time.time()))+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
