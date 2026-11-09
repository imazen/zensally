#!/usr/bin/env python3
"""Freeze complete matched extent fine-tuning, including fit/development audit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifact_identity import sha
from freeze_florence_evidence import freeze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('auxiliary',type=Path)
    args = p.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    for folder,count in [('extent-giou-fits',9),('extent-joint-fits',6)]:
        progress = json.loads((args.root/folder/'progress.json').read_text())
        if progress['pid'] is not None or progress['stage'] != 'complete' or progress['completed_runs'] != count:
            raise ValueError('verified completed fits required')
    analysis = json.loads((args.root/'analysis.json').read_text())
    audit = json.loads((args.root/'generalization.json').read_text())
    if analysis['completed_fits'] != 15 or len(audit['runs']) != 30 or analysis['selected_replay_max_map_edge_difference'] != 0:
        raise ValueError('complete matched numerical audit required')
    if sha(args.auxiliary) != '463e4884be5ada2128d08b9a755cf836d23e116059e6716390d64c0712b768a3':
        raise ValueError('registered original source cache differs')
    (args.root/'inputs').mkdir()
    shutil.copy2(args.auxiliary,args.root/'inputs/auxiliary.pt')
    files = [Path('tools/composition')/n for n in ['prepare_florence_finetune.py',
        'fit_extent_images.py','audit_extent_generalization.py','analyze_extent_finetune.py','freeze_extent_finetune.py']]
    files += [Path('training')/n for n in ['crop_fit.py','crop_regions.py','crop_ranker.py',
        'model_v3ds.py','composition_head.py','composition_extent_image.py']]
    files += [Path('docs/experiments/FLORENCE_EXTENT_FINETUNE.md'),Path('benchmarks/florence_extent_finetune_2026-11-09.md')]
    producers = {}
    for path in files:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        producers[str(path)] = subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (args.root/'fit-logs').mkdir()
    for pattern in ['zensally-florence-finetune*20261109.log','zensally-extent-*20261109.log']:
        for path in Path('/home/lilith/tmp').glob(pattern):
            shutil.copy2(path,args.root/'fit-logs'/path.name)
    freeze(args.root,dict(completed_fits=15,audited_checkpoints=30,additional_source_producers=producers,
        target_packet_sha256='684d74cf1070af1378dc49b45bc14104c1e7be93356161b22dfbb0548ada3b3c',
        training_admission='same48 fit-only weak unions; eight development targets select heads',
        teacher_inference_calls=0))


if __name__ == '__main__':
    main()
