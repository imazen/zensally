#!/usr/bin/env python3
"""Check batch1 deployment against original frozen batched dev/TEST choices."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from artifact_identity import sha
from export_unsplash_ranker import LocalCropRanker, session


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'export', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.study/'SHA256SUMS.json').read_text())
    identities = {v['path']:v['sha256'] for v in manifest}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35)
    weights = args.study/'fits/trainable-image-seed911/best.pt'
    if sha(weights) != 'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':
        raise ValueError('preselected student differs')
    model = LocalCropRanker().cuda().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['model'])
    encoder, query = [session(args.export/(name+'.onnx')) for name in ['encoder', 'query']]
    qualification = json.loads((args.export/'qualification.json').read_text())
    for name in ['encoder', 'query']:
        if sha(args.export/(name+'.onnx')) != qualification['exports'][name]['sha256']:
            raise ValueError('qualified graphs changed')
    totals = {name:dict(requests=0, changed_choices=0, gate_violations=0, max_edge_map_pixels=0.)
              for name in ['cuda-batch1', 'ort-batch1']}
    changed = []
    for cache_file, prediction_file, role in [
            ('cache/cache.pt', 'fits/trainable-image-seed911/best-predictions.json', 'dev'),
            ('test-cache/cache.pt', 'test-evaluation/trainable-image-seed911.json', 'test')]:
        for relative in [cache_file, prediction_file]:
            if sha(args.study/relative) != identities[relative]:
                raise ValueError('frozen original inputs/predictions differ')
        cache = torch.load(args.study/cache_file, map_location='cpu', weights_only=False)
        original = {(r['id'], r['query']):r['selected'] for r in json.loads((args.study/prediction_file).read_text())}
        sources = [(i,s) for i,s in enumerate(cache['sources']) if s['role'] == role]
        if len(sources) != 25 or len(original) != 550:
            raise ValueError('all25 frozen sources/550 requests per role required')
        for i, source in sources:
            image = cache['image'][i:i+1]
            features = model.feature_maps(image.cuda())
            f8, f16 = encoder.run(None, dict(image=image.numpy()))
            for q in range(22):
                p, ratio_q = map(int, cache['ratios'][q//2].split(':'))
                values = dict(boxes=cache['boxes'][i:i+1, q//2].numpy(),
                    source_aspect=np.array([source['native_size'][0]/source['native_size'][1]], np.float32),
                    target_aspect=np.array([p/ratio_q], np.float32), mode=np.array([q % 2], np.float32),
                    content=cache['content'][i:i+1].numpy())
                cuda = model.scores(features, *(torch.from_numpy(v).cuda() for v in values.values())).cpu().numpy()[0]
                ort = query.run(None, dict(f8=f8, f16=f16, **values))[0][0]
                expected = original[(source['id'], q)]
                eligible = cache['eligible'][i, q].numpy()
                for name, raw in [('cuda-batch1', cuda), ('ort-batch1', ort)]:
                    if not np.isfinite(raw).all():
                        raise ValueError('nonfinite batch1 scores')
                    selected = int(np.where(eligible, raw, -np.inf).argmax())
                    boxes = values['boxes'][0]
                    motion = float(np.abs(boxes[selected]-boxes[expected]).max()*128)
                    result = totals[name]
                    result['requests'] += 1
                    result['changed_choices'] += selected != expected
                    result['gate_violations'] += motion > 3
                    result['max_edge_map_pixels'] = max(result['max_edge_map_pixels'], motion)
                    if selected != expected:
                        changed.append(dict(id=source['id'], role=role, query=q, runtime=name,
                            original=expected, selected=selected, max_edge_map_pixels=motion))
    report = dict(totals=totals, changed=changed,
        reference='original frozen25-source evaluations: encoder chunks16/9,25-source candidate-query batch',
        source_roles_changed=False, training_updated=False, model_reselected=False,
        teacher_reference_scores_used=False, face_coverage_certified=False)
    (args.output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if any(v['gate_violations'] for v in totals.values()):
        raise ValueError('batch1 port exceeds the unchanged crop-edge gate')


if __name__ == '__main__':
    main()
