#!/usr/bin/env python3
"""Freeze117 pre-existing synthetic-person ROI inputs with inherited roles."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from artifact_identity import sha
from roi_geometry import person_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime', 'roles', 'resizer', 'encoder', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('owned resizer differs')
    if sha(args.encoder) != '626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':
        raise ValueError('owned encoder differs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 8*1024**3:
        raise ValueError('need8GiB disk headroom')
    manifest_path = args.runtime/'manifest.json'
    results_path = args.runtime/'reference-eval/int8-efficientdet_lite0/results.json'
    manifest = json.loads(manifest_path.read_text())
    roles = json.loads(args.roles.read_text())
    sources = roles['sources']
    if (len(sources) != 56 or sum(s['use'] == 'fit' for s in sources) != 48
            or sum(s['use'] == 'dev' for s in sources) != 8):
        raise ValueError('original56source roles required')
    if {s['family_id'] for s in sources if s['use'] == 'fit'} & {s['family_id'] for s in sources if s['use'] == 'dev'}:
        raise ValueError('family roles overlap')
    results = {r['id']: r for r in json.loads(results_path.read_text())['rows']}
    images = {s['id']: s for s in manifest['images']}
    models = [next(m for m in manifest['models'] if m['id'] == name) for name in
              ['float16-blaze_face_full_range_sparse', 'float16-pose_landmarker_lite']]
    for model in models:
        if sha(Path(model['path'])) != model['sha256']:
            raise ValueError('model changed')
    args.output.mkdir(parents=True, exist_ok=False)
    views, total = [], 0
    for source in sources:
        image = images[source['id']]
        raw = Path(image['rgb'])
        width, height = image['width'], image['height']
        if not image['synthetic'] or image['rgb_sha256'] != source['rgb_sha256'] or sha(raw) != source['rgb_sha256']:
            raise ValueError('synthetic source/pixels differ')
        specs, metadata = [], []
        for index, detection in enumerate(results[source['id']]['result']['detections']):
            if not any(c['name'] == 'person' for c in detection['categories']):
                continue
            _, bounds = person_views(detection['xywh'], [width, height], [width, height])[0]
            l, t, r, b = bounds
            scale = min(1, 768/max(r-l, b-t))
            ow, oh = max(1, round((r-l)*scale)), max(1, round((b-t)*scale))
            specs.append(','.join(map(str, [l, t, r-l, b-t, ow, oh])))
            metadata.append(dict(id=source['id']+f'-person{index:02}', source_id=source['id'],
                family_id=source['family_id'], role=source['use'], kind='person',
                native_width=width, native_height=height, native_bounds=bounds,
                width=ow, height=oh, detection=detection, source_rgb_sha256=source['rgb_sha256']))
        if not specs:
            continue
        with raw.open('rb') as stream:
            pixels = subprocess.check_output([str(args.resizer), str(width), str(height), str(width*3),
                                              ';'.join(specs)], stdin=stream)
        if len(pixels) != sum(r['width']*r['height']*3 for r in metadata):
            raise ValueError('owned resize output differs')
        offset = 0
        for row in metadata:
            size = row['width']*row['height']*3
            rgb, png = args.output/(row['id']+'.rgb'), args.output/(row['id']+'.png')
            rgb.write_bytes(pixels[offset:offset+size])
            subprocess.run([str(args.encoder), '--from-rgb', str(row['width']), str(row['height']),
                            str(rgb), str(png)], check=True)
            row.update(rgb=str(rgb), rgb_sha256=sha(rgb), png=str(png), png_sha256=sha(png))
            views.append(row)
            offset += size
            total += size+png.stat().st_size
        (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), stage='rendering',
            completed_views=len(views), required_views=117, last_checked_unix=time.time()))+'\n')
    if len(views) != 117 or sum(v['role'] == 'fit' for v in views) != 94 or total > 512*1024**2:
        raise ValueError('registered view coverage/storage bound differs')
    doc = dict(study='synthetic-person-rois-20261108-v1', views=views, sources=sources,
        models=models, roles_sha256=sha(args.roles), detector_results_sha256=sha(results_path),
        runtime_manifest_sha256=sha(manifest_path), payload_bytes=total,
        training_eligible=False, geometry_target_admission='suitability first; fit protocol required',
        face_coverage_certified=False, diagnostic_maps_saved=False)
    (args.output/'manifest.json').write_text(json.dumps(doc, indent=2)+'\n')
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete',
        completed_views=117, required_views=117, last_checked_unix=time.time()))+'\n')
    print(json.dumps(dict(views=len(views), payload_bytes=total)))


if __name__ == '__main__':
    main()
