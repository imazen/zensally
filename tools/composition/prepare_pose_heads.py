#!/usr/bin/env python3
"""Freeze pose head additions and audit existing crop rectangles, without labels."""
import argparse
import json
from pathlib import Path

from artifact_identity import sha
from pose_head_bounds import head_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen_screen', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest_path = args.frozen_screen/'person-roi-inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    poses_path = args.frozen_screen/'person-roi-results/float16-pose_landmarker_lite.json'
    poses = json.loads(poses_path.read_text())
    if poses['manifest_sha256'] != sha(manifest_path):
        raise ValueError('pose/input identity differs')
    views = {r['id']: r for r in manifest['views']}
    sources, skipped, old_violations = [], [], []
    for source_id in ['11048', '11214']:
        bank_path = args.frozen_screen/'crop-proposals'/(source_id+'.json')
        bank = json.loads(bank_path.read_text())
        regions = []
        for row in poses['rows']:
            if row['source_id'] != source_id or row['kind'] != 'person':
                continue
            view = views[row['id']]
            if (view['native_width'], view['native_height']) != (bank['width'], bank['height']):
                raise ValueError('ROI/bank source shape differs')
            for index, points in enumerate(row['projected_native']):
                box, reason = head_bounds(points, bank['width'], bank['height'])
                record = dict(view_id=row['id'], pose_index=index, reason=reason)
                if box is None:
                    skipped.append(record)
                else:
                    regions.append(dict(record, pixel_box=box))
        for task in bank['tasks']:
            for selection in task['selected']:
                index = selection['detected_head_guarded_candidate']
                if index is None:
                    continue
                l, t, r, b = bank_box = task['candidates'][index]['pixel_box']
                clipped = [region['view_id'] for region in regions if not (
                    l <= region['pixel_box'][0] and t <= region['pixel_box'][1]
                    and r >= region['pixel_box'][2] and b >= region['pixel_box'][3])]
                if clipped:
                    old_violations.append(dict(id=source_id, ratio=task['ratio'], mode=selection['mode'],
                        old_pixel_box=bank_box, clipped_pose_regions=clipped))
        sources.append(dict(id=source_id, bank_sha256=sha(bank_path), added_regions=regions))
    args.output.mkdir(parents=True, exist_ok=False)
    doc = dict(sources=sources, skipped=skipped, old_violations=old_violations,
        roi_manifest_sha256=sha(manifest_path), pose_results_sha256=sha(poses_path),
        face_coverage_certified=False, training_eligible=False, deployable=False)
    (args.output/'regions.json').write_text(json.dumps(doc, indent=2)+'\n')
    print(json.dumps(dict(sources=2, added_regions=sum(len(s['added_regions']) for s in sources),
        skipped=len(skipped), old_offered_crops_clipping_estimated_heads=len(old_violations))))


if __name__ == '__main__':
    main()
