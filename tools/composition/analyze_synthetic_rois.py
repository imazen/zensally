#!/usr/bin/env python3
"""Measure local head-target support before admitting any ROI student fit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from artifact_identity import sha
from pose_head_bounds import head_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    manifest_path = args.root/'inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    oracle = json.loads((args.root/'input-oracle.json').read_text())
    if len(oracle['checks']) != 117 or not all(c['png_rgb_exact'] for c in oracle['checks']):
        raise ValueError('complete independent input readback required')
    pose_path = args.root/'teacher/float16-pose_landmarker_lite.json'
    face_path = args.root/'teacher/float16-blaze_face_full_range_sparse.json'
    pose, face = [json.loads(p.read_text()) for p in [pose_path, face_path]]
    if any(d['manifest_sha256'] != sha(manifest_path) for d in [pose, face]):
        raise ValueError('teacher/input identities differ')
    ids = [v['id'] for v in manifest['views']]
    if any([r['id'] for r in d['rows']] != ids for d in [pose, face]):
        raise ValueError('teacher view coverage/order differs')
    rows, totals = [], Counter()
    support = defaultdict(lambda: dict(rois=0, sources=set(), families=set()))
    for view, poses, faces in zip(manifest['views'], pose['rows'], face['rows'], strict=True):
        regions, reasons = [], []
        l, t, r, b = view['native_bounds']
        for index, landmarks in enumerate(poses['projected_native']):
            box, reason = head_bounds(landmarks, view['native_width'], view['native_height'])
            if box is None:
                reasons.append(reason)
            elif not (l <= box[0] < box[2] <= r and t <= box[1] < box[3] <= b):
                reasons.append('pose-derived head extends outside the input ROI')
            else:
                regions.append(dict(pose_index=index, pixel_box=box))
        target = None
        if regions and not reasons:
            union = [min(h['pixel_box'][0] for h in regions), min(h['pixel_box'][1] for h in regions),
                     max(h['pixel_box'][2] for h in regions), max(h['pixel_box'][3] for h in regions)]
            target = [(union[0]-l)/(r-l), (union[1]-t)/(b-t), (union[2]-l)/(r-l), (union[3]-t)/(b-t)]
            part = support[view['role']]
            part['rois'] += 1
            part['sources'].add(view['source_id'])
            part['families'].add(view['family_id'])
        raw_faces = [f['visible_native_box'] for f in faces['projected_native'] if f['visible_native_box']]
        unmatched = sum(not any(max(f[0], h['pixel_box'][0]) < min(f[2], h['pixel_box'][2])
                            and max(f[1], h['pixel_box'][1]) < min(f[3], h['pixel_box'][3])
                            for h in regions) for f in raw_faces)
        totals.update(rois=1, pose_instances=len(poses['projected_native']),
            face_instances=len(raw_faces), usable_targets=int(target is not None),
            no_pose=int(not poses['projected_native']), uncertain_poses=len(reasons),
            faces_unmatched_to_available_pose_head=unmatched)
        rows.append(dict(id=view['id'], source_id=view['source_id'], family_id=view['family_id'],
            role=view['role'], pose_regions=regions, unavailable_reasons=reasons,
            pose_count=len(poses['projected_native']), face_count=len(raw_faces),
            unmatched_faces=unmatched, target_roi_normalized=target,
            training_eligible=False, human_rated=False, face_coverage_certified=False,
            rgb_sha256=view['rgb_sha256'], source_rgb_sha256=view['source_rgb_sha256']))
    counts = {role: dict(rois=v['rois'], sources=len(v['sources']), families=len(v['families']))
              for role, v in support.items()}
    fit_gate = counts.get('fit', {}).get('families', 0) >= 12 and counts.get('dev', {}).get('families', 0) >= 3
    report = dict(totals=dict(totals), support=counts, minimum_fit_family_support_met=fit_gate,
        requires_registered_fit_protocol=True, teacher_quality_independently_certified=False,
        input_manifest_sha256=sha(manifest_path), pose_results_sha256=sha(pose_path),
        face_results_sha256=sha(face_path), rows=rows, training_eligible=False)
    with (args.root/'target-suitability.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(totals=report['totals'], support=counts, minimum_fit_family_support_met=fit_gate)))


if __name__ == '__main__':
    main()
