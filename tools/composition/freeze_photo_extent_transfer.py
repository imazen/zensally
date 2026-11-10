#!/usr/bin/env python3
"""Freeze the complete negative photographic-feature transfer comparison."""
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
    p.add_argument('weights',type=Path)
    args = p.parse_args()
    if (args.root/'SHA256SUMS.json').exists():
        raise ValueError('packet already frozen')
    if sha(args.weights) != 'c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df':
        raise ValueError('preselected model identity differs')
    progress = json.loads((args.root/'extent-fits/progress.json').read_text())
    analysis = json.loads((args.root/'analysis.json').read_text())
    if progress['pid'] is not None or progress['completed_runs'] != 9 or not analysis['matched_aspect_controls_exact']:
        raise ValueError('completed matched comparison required')
    files = [Path('tools/composition')/n for n in ['cache_photo_student_extents.py',
        'analyze_photo_student_extents.py','freeze_photo_extent_transfer.py']]
    files += [Path('training')/n for n in ['local_rank_distillation.py','crop_fit.py',
        'crop_regions.py','crop_ranker.py','model_v3ds.py','composition_head.py']]
    files += [Path('docs/experiments/PHOTO_STUDENT_EXTENT_TRANSFER.md'),
              Path('benchmarks/photo_student_extent_transfer_2026-11-09.md')]
    producers = {}
    for path in files:
        target = args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        producers[str(path)] = subprocess.check_output(['git','log','-1','--format=%H','--',str(path)],text=True).strip()
    (args.root/'model').mkdir()
    shutil.copy2(args.weights,args.root/'model/photographic-student.pt')
    (args.root/'transfer-logs').mkdir()
    for path in Path('/home/lilith/tmp').glob('zensally-photo-extent*20261109.log'):
        shutil.copy2(path,args.root/'transfer-logs'/path.name)
    freeze(args.root,dict(completed_fits=9,encoder_frozen=True,transfer_source_producers=producers,
        target_packet_sha256='684d74cf1070af1378dc49b45bc14104c1e7be93356161b22dfbb0548ada3b3c',
        photographic_rank_packet_sha256='d2a9e3c1b2f67a0db01088c52b865895467b1e86c6fbc5588f9e2a15a49b1b8e',
        teacher_inference_calls=0,training_admission='same48 fit-only weak unions; eight development targets select heads'))


if __name__ == '__main__':
    main()
