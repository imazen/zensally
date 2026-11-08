#!/usr/bin/env python3
"""Reference-only pose/face calls on frozen person crops, without auto-admission."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime-eval'))
from mediapipe_probe import load, make, summarize
from artifact_identity import sha
from roi_geometry import project_face


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text())
    if len(doc['views']) != 18 or len(doc['models']) != 3:
        raise ValueError('registered bound differs')
    args.output.mkdir(parents=True, exist_ok=False)
    counts = []
    for model in doc['models']:
        if sha(Path(model['path'])) != model['sha256']:
            raise ValueError('model changed')
        runner, call, kind = make(model)
        if kind not in ['faces', 'pose']:
            raise ValueError('coordinate outputs only')
        rows = []
        try:
            for view in doc['views']:
                result = summarize(call(load(view)), kind, view, args.output)
                projected = []
                if kind == 'faces':
                    for detection in result['detections']:
                        projected.append(dict(**project_face(detection['xywh'], view['native_bounds'],
                            [view['width'], view['height']], [view['native_width'], view['native_height']]),
                            categories=detection['categories']))
                else:
                    left, top, right, bottom = view['native_bounds']
                    for landmarks in result['landmarks']:
                        projected.append([dict(x=left+p['x']*(right-left), y=top+p['y']*(bottom-top),
                            visibility=p['visibility'], presence=p['presence']) for p in landmarks])
                rows.append(dict(id=view['id'], source_id=view['source_id'], kind=view['kind'],
                                 result=result, projected_native=projected))
                counts.append(dict(model=model['id'], id=view['id'], count=len(projected)))
        finally:
            runner.close()
        (args.output/(model['id']+'.json')).write_text(json.dumps(dict(model=model,
            manifest_sha256=sha(args.manifest), rows=rows), indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(calls=len(counts), counts=counts,
        training_eligible=False, human_rated=False, face_coverage_certified=False), indent=2)+'\n')
    print(json.dumps(dict(calls=len(counts), positive_calls=sum(c['count'] > 0 for c in counts))))


if __name__ == '__main__':
    main()
