#!/usr/bin/env python3
"""Make registered person ROI inputs with owned RGB and the pinned resizer."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from artifact_identity import sha
from roi_geometry import person_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['screen', 'acquired', 'runtime_manifest', 'resizer', 'encoder', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    expected = {'resizer': 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041',
                'encoder': '626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d'}
    for key, digest in expected.items():
        if sha(getattr(args, key)) != digest:
            raise ValueError('owned executable differs: '+key)
    inputs = json.loads((args.screen/'inputs.json').read_text())
    sources = json.loads((args.acquired/'sources.json').read_text())
    results = json.loads((args.screen/'objects-efficientdet/results.json').read_text())
    runtime = json.loads(args.runtime_manifest.read_text())
    models = [next(m for m in runtime['models'] if m['id'] == name) for name in
              ['float16-blaze_face_full_range_sparse', 'float16-blaze_face_short_range',
               'float16-pose_landmarker_lite']]
    for model in models:
        if sha(Path(model['path'])) != model['sha256']:
            raise ValueError('model differs')
    args.output.mkdir(parents=True, exist_ok=False)
    views, identities, total = [], [], 0
    for source_id, expected_count in [('11048', 4), ('11214', 5)]:
        source = next(s for s in sources if s['id'] == source_id)
        preview = next(s for s in inputs['images'] if s['id'] == source_id)
        result = next(s for s in results['rows'] if s['id'] == source_id)
        if source['training_eligible'] or source['use'] != 'screening' or source['bucket'] != 'train':
            raise ValueError('source is not unadmitted TRAIN')
        path = Path(source['directory'])/'source.rgb'
        nw, nh = source['width'], source['height']
        if path.stat().st_size != nw*nh*3 or sha(path) != source['rgb_sha256']:
            raise ValueError('owned native RGB differs')
        detections = [(i, d) for i, d in enumerate(result['result']['detections'])
                      if any(c['name'] == 'person' for c in d['categories'])]
        if len(detections) != expected_count:
            raise ValueError('registered detections differ')
        specs, metadata = [], []
        for index, detection in detections:
            for kind, box in person_views(detection['xywh'], [preview['width'], preview['height']], [nw, nh]):
                left, top, right, bottom = box
                width, height = right-left, bottom-top
                scale = min(1, 768/max(width, height))
                ow, oh = max(1, round(width*scale)), max(1, round(height*scale))
                specs.append(','.join(map(str, [left, top, width, height, ow, oh])))
                metadata.append(dict(id=f'{source_id}-{index}-{kind}', source_id=source_id,
                    detection_index=index, detection=detection, kind=kind, native_bounds=box,
                    native_width=nw, native_height=nh, width=ow, height=oh))
        with path.open('rb') as stream:
            rendered = subprocess.check_output([str(args.resizer), str(nw), str(nh), str(nw*3),
                                                ';'.join(specs)], stdin=stream)
        if len(rendered) != sum(r['width']*r['height']*3 for r in metadata):
            raise ValueError('owned resizer output length differs')
        offset = 0
        for row in metadata:
            size = row['width']*row['height']*3
            rgb = rendered[offset:offset+size]
            raw, png = args.output/(row['id']+'.rgb'), args.output/(row['id']+'.png')
            raw.write_bytes(rgb)
            subprocess.run([str(args.encoder), '--from-rgb', str(row['width']), str(row['height']),
                            str(raw), str(png)], check=True, stdout=subprocess.PIPE)
            row.update(rgb=str(raw), rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                       png=str(png), png_sha256=sha(png))
            views.append(row)
            offset += size
            total += size+png.stat().st_size
        identities.append(dict(id=source_id, family_id=source['family_id'], native_rgb_sha256=sha(path),
            baseline_face_sha256=sha(args.screen/'geometry'/(source_id+'-faces.json')),
            baseline_pose_sha256=sha(args.screen/'geometry'/(source_id+'-pose.json'))))
    if len(views) != 18 or total > 64*1024*1024:
        raise ValueError('registered view count/storage budget differs')
    document = dict(views=views, models=models, sources=identities, view_payload_bytes=total,
        inputs_sha256=sha(args.screen/'inputs.json'),
        detector_results_sha256=sha(args.screen/'objects-efficientdet/results.json'),
        executables=expected, training_eligible=False, face_coverage_certified=False,
        diagnostic_maps_saved=False, paid_model_api_calls=0)
    (args.output/'manifest.json').write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(dict(views=len(views), view_payload_bytes=total)))


if __name__ == '__main__':
    main()
