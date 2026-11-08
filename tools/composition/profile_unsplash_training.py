#!/usr/bin/env python3
"""Bound a diagnostic copy of the exact fitter with profiler-only hooks."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from artifact_identity import sha
import fit_unsplash_ranks as fitter


class CaptureComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'checkpoint', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cgroup = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::'))
    limit_path = Path('/sys/fs/cgroup')/cgroup.lstrip('/')/'memory.max'
    limit = limit_path.read_text().strip()
    if limit == 'max' or int(limit) > 4*1024**3:
        raise ValueError('profile target requires its own verified4GiB-or-lower cgroup cap')
    (args.output/'TARGET_SCOPE.json').write_text(json.dumps(dict(pid=os.getpid(), cgroup=cgroup,
        memory_max_bytes=int(limit), scope_verified_before_fitting=True), indent=2)+'\n')
    state = dict(steps=0, active_optimizer=None, capture_started=False, capture_stopped=False)
    original_init = torch.optim.AdamW.__init__
    original_features = fitter.LocalCropRanker.feature_maps
    original_scores = fitter.LocalCropRanker.scores
    original_evaluate = fitter.evaluate
    original_backward = torch.Tensor.backward

    def initialize(optimizer, *values, **keywords):
        original_init(optimizer, *values, **keywords)
        count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        if count > 100000 and state['active_optimizer'] is None:
            state['active_optimizer'] = id(optimizer)
            def after(current, _args, _kwargs):
                state['steps'] += 1
                if state['steps'] == 220:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    state['capture_stopped'] = True
                    raise CaptureComplete()
            optimizer.register_step_post_hook(after)

    def features(model, image):
        if model.training and model.arm == 'trainable-image' and state['steps'] == 20 and not state['capture_started']:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            state['capture_started'] = True
        label = 'training.encoder' if model.training else 'development.encoder'
        with torch.cuda.nvtx.range(label):
            return original_features(model, image)

    def scores(model, *values, **keywords):
        with torch.cuda.nvtx.range('training.query' if model.training else 'development.query'):
            return original_scores(model, *values, **keywords)

    def evaluate(*values, **keywords):
        with torch.cuda.nvtx.range('development'):
            return original_evaluate(*values, **keywords)

    def backward(tensor, *values, **keywords):
        with torch.cuda.nvtx.range('training.backward'):
            return original_backward(tensor, *values, **keywords)

    torch.optim.AdamW.__init__ = initialize
    fitter.LocalCropRanker.feature_maps = features
    fitter.LocalCropRanker.scores = scores
    fitter.evaluate = evaluate
    torch.Tensor.backward = backward
    sys.argv = ['fit_unsplash_ranks.py', str(args.cache), str(args.checkpoint), str(args.output/'diagnostic-fits')]
    try:
        fitter.main()
        raise RuntimeError('profiler bound did not fire')
    except CaptureComplete:
        if not state['capture_started'] or not state['capture_stopped'] or state['steps'] != 220:
            raise ValueError('exact warmup/capture boundary differs')
    finally:
        torch.optim.AdamW.__init__ = original_init
        fitter.LocalCropRanker.feature_maps = original_features
        fitter.LocalCropRanker.scores = original_scores
        fitter.evaluate = original_evaluate
        torch.Tensor.backward = original_backward
    report = dict(pid=None, completed=True, kind='bounded diagnostic copy, not a complete scientific fit',
        original_scientific_fits_completed=12, source_cache_sha256=sha(args.cache),
        initializer_sha256=sha(args.checkpoint), fitter_source_sha256=sha(Path(fitter.__file__)),
        warmup_optimizer_steps=20, captured_optimizer_steps=200,
        capture_started_before_training_encoder=True, capture_stopped_after_optimizer=True,
        forward_math_changed=False, quality_model_selected=False, test_sources_opened=0,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (args.output/'PROFILE_COMPLETION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
