#!/usr/bin/env python3
"""Export the pre-TEST selected student; qualify its own CPU/runtime arithmetic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from artifact_identity import sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from local_rank_distillation import LocalCropRanker


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.feature_maps(image)


class Query(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, f8, f16, boxes, source_aspect, target_aspect, mode, content):
        return self.model.scores((f8, f16), boxes, source_aspect, target_aspect, mode, content)


class Complete(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, boxes, source_aspect, target_aspect, mode, content):
        return self.model.scores(self.model.feature_maps(image), boxes,
            source_aspect, target_aspect, mode, content)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['fit_cache', 'test_cache', 'plan', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--gather-pool', action='store_true')
    args = parser.parse_args()
    for path, digest in [(args.fit_cache, '2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'),
                         (args.test_cache, '2f5a2d2fc1b61ce34d5181746bd12d46ad9973ae19f2ee0db4b1e7c2c5f16f6f'),
                         (args.plan, '6358383b2b7b2e1a3386334168fd9b457c9c2c17b2e37b96b74b003520696f45')]:
        if sha(path) != digest:
            raise ValueError('frozen input identity differs')
    plan = json.loads(args.plan.read_text())
    selected = plan['candidate_for_export']
    binding = next(b for b in plan['models'] if b['arm'] == selected['arm'] and b['seed'] == selected['seed'])
    weights = Path(binding['path'])
    if sha(weights) != selected['weights_sha256']:
        raise ValueError('preselected weights changed')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35)
    state = torch.load(weights, map_location='cpu', weights_only=False)
    model = LocalCropRanker().eval()
    model.load_state_dict(state['model'])
    gpu = LocalCropRanker().cuda().eval()
    gpu.load_state_dict(state['model'])
    caches = [torch.load(p, map_location='cpu', weights_only=False) for p in [args.fit_cache, args.test_cache]]
    cache = caches[0]
    sample = (cache['image'][:1], cache['boxes'][:1, 0], torch.ones(1), torch.ones(1),
              torch.zeros(1), cache['content'][:1])
    encoded = model.feature_maps(sample[0])
    names = ['image', 'boxes', 'source_aspect', 'target_aspect', 'mode', 'content']
    query_names = ['f8', 'f16']+names[1:]
    exports = [('encoder', Encoder(model), (sample[0],), ['image'], ['f8', 'f16']),
               ('query', Query(model), (*encoded, *sample[1:]), query_names, ['scores']),
               ('complete', Complete(model), sample, names, ['scores'])]
    identities = {}
    before_export = {key:value.clone() for key,value in model.state_dict().items()}
    for name, module, values, inputs, outputs in exports:
        # The legacy exporter restores the wrapper's training flag recursively.
        # A fresh wrapper defaults to train=True even around an eval child.
        module.eval()
        path = args.output/(name+'.onnx')
        torch.onnx.export(module, values, str(path), input_names=inputs, output_names=outputs,
                          opset_version=18, dynamo=False, external_data=False)
        if args.gather_pool and name != 'encoder':
            from gather_grid import lower_student_grid_samples
            lower_student_grid_samples(path)
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph, full_check=True)
        identities[name] = dict(bytes=path.stat().st_size, sha256=sha(path),
            nodes=len(graph.graph.node), operators=dict(Counter(n.op_type for n in graph.graph.node)))
    if any(m.training for m in model.modules()) or any(
            not torch.equal(value, before_export[key]) for key,value in model.state_dict().items()):
        raise ValueError('export mutated evaluation mode or model/BN state')
    sessions = {name:session(args.output/(name+'.onnx')) for name in identities}
    records = []
    totals = {name:dict(queries=0, changed_choices=0, gate_violations=0,
        max_score_absolute_error=0., max_edge_map_pixels=0.) for name in ['torch-cpu', 'ort-split', 'ort-complete']}
    for cache in caches:
        for i, source in enumerate(cache['sources']):
            image = cache['image'][i:i+1]
            features = model.feature_maps(image)
            gpu_features = gpu.feature_maps(image.cuda())
            ort_features = sessions['encoder'].run(None, {'image':image.numpy()})
            for q in range(22):
                ratio = cache['ratios'][q//2].split(':')
                values = (cache['boxes'][i:i+1, q//2],
                    torch.tensor([source['native_size'][0]/source['native_size'][1]]),
                    torch.tensor([int(ratio[0])/int(ratio[1])]), torch.tensor([q % 2], dtype=torch.float32),
                    cache['content'][i:i+1])
                reference = gpu.scores(gpu_features, *(v.cuda() for v in values)).cpu().numpy()[0]
                cpu = model.scores(features, *values).numpy()[0]
                feed = {key:value.numpy() for key,value in zip(names[1:], values)}
                split = sessions['query'].run(None, dict(f8=ort_features[0], f16=ort_features[1], **feed))[0][0]
                complete = sessions['complete'].run(None, dict(image=image.numpy(), **feed))[0][0]
                eligible = cache['eligible'][i, q].numpy()
                chosen = int(np.where(eligible, reference, -np.inf).argmax())
                boxes = values[0].numpy()[0]
                for name, result in [('torch-cpu', cpu), ('ort-split', split), ('ort-complete', complete)]:
                    if not np.isfinite(result).all():
                        raise ValueError('nonfinite runtime score')
                    selected_index = int(np.where(eligible, result, -np.inf).argmax())
                    motion = float(np.abs(boxes[selected_index]-boxes[chosen]).max()*128)
                    error = float(np.abs(result-reference).max())
                    row = totals[name]
                    row['queries'] += 1
                    row['changed_choices'] += selected_index != chosen
                    row['gate_violations'] += motion > 3
                    row['max_score_absolute_error'] = max(row['max_score_absolute_error'], error)
                    row['max_edge_map_pixels'] = max(row['max_edge_map_pixels'], motion)
                    if selected_index != chosen:
                        records.append(dict(id=source['id'], role=source['role'], query=q, runtime=name,
                            reference=chosen, selected=selected_index, max_edge_map_pixels=motion))
            print(json.dumps(dict(completed_sources=i+1, role=source['role'], totals=totals)), flush=True)
    result = dict(weights_sha256=sha(weights), selected_before_test=True, parameters=sum(p.numel() for p in model.parameters()),
        exports=identities, runtime_gate=totals, changed_choices=records,
        gate='per-axis normalized source coordinates times128; at most3 per rectangle edge',
        reference='same frozen student FP32 Torch CUDA, TF32 disabled; batch1',
        original_test_reference_scores_used=False, source_roles_changed=False,
        human_artistic_quality_measured=False, face_coverage_certified=False, model_promoted=False,
        torch_version=torch.__version__, onnx_version=onnx.__version__, ort_version=ort.__version__,
        pooling='fixed finite border bilinear gather' if args.gather_pool else 'original GridSample',
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (args.output/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    if any(r['gate_violations'] for r in totals.values()):
        raise ValueError('runtime crop-edge gate failed; retained all evidence')
    print(json.dumps(dict(runtime_gate=totals, exports=identities, parameters=result['parameters'])))


if __name__ == '__main__':
    main()
