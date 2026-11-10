#!/usr/bin/env python3
"""Bind fixed selected model states before supplementary weak-reference inference."""
import argparse
import json
from pathlib import Path
import subprocess

from artifact_identity import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','studies','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    if sha(args.cohort)!='a52144833a5824386577f63626e599f8dc602f8b77ee9d4d7f0c183574cebf70':
        raise ValueError('registered supplementary source roles differ')
    directories = [
        ('small-mse','florence-primary56','extent-fits',9),
        ('focal-mse','florence-primary56','focalnet/extent-fits',9),
        ('photo-mse','florence-photo-transfer','extent-fits',9),
        ('small-giou','florence-finetune','extent-giou-fits',9),
        ('small-trained','florence-finetune','extent-joint-fits',6),
        ('small-augmented','florence-extent-augmentation','extent-joint-augmented-fits',6)]
    models = []
    for name,study,relative,count in directories:
        root = args.studies/study/'2026-11-09'
        directory = root/relative
        runs = json.loads((directory/'summary.json').read_text())
        if len(runs)!=count:
            raise ValueError('complete fixed model matrix required')
        for run in runs:
            weights = directory/(run['arm']+'-seed'+str(run['seed']))/'best.pt'
            models.append(dict(study=name,arm=run['arm'],seed=run['seed'],selected_epoch=run['best_epoch'],
                checkpoint=str(weights),checkpoint_sha256=sha(weights),
                original_source_packet_sha256=sha(root/'SHA256SUMS.json'),
                original_selection=run['best_dev']))
    if len(models)!=48:
        raise ValueError('all48 prior selected checkpoints required')
    args.output.mkdir(parents=True,exist_ok=False)
    result = dict(build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        cohort_manifest_sha256=sha(args.cohort),models=models,required_models=48,
        new_model_selection=False,new_gradients=False,new_test_inference=False,
        paid_model_api_calls=0,reference_targets_generated=False,face_coverage_certified=False)
    path = args.output/'evaluation-plan.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(frozen_models=48,supplementary_development_sources=15,plan_sha256=sha(path))))


if __name__ == '__main__':
    main()
