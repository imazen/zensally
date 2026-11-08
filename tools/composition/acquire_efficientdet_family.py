#!/usr/bin/env python3
"""Acquire official sample metadata models for the requested Lite1/Lite2 screen."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    models = []
    for index in [0, 1, 2]:
        url = ('https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/'
               f'object_detection/android/lite-model_efficientdet_lite{index}_detection_metadata_1.tflite')
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(64*1024**2+1)
        if len(data) > 64*1024**2 or data[4:8] != b'TFL3':
            raise ValueError('bounded TFLite model required')
        path = args.output/f'tf-efficientdet_lite{index}.tflite'
        path.write_bytes(data)
        models.append(dict(id=f'tf-efficientdet_lite{index}', path=str(path), url=url,
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            source_download_script_git_blob='f97894603671f8611750157541a9b8eb97d630fd',
            source_url='https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/download_models.gradle'))
        (args.output/'models.json').write_text(json.dumps(models, indent=2)+'\n')
        print(json.dumps(models[-1]), flush=True)
    old = json.loads(Path('/mnt/v/output/zensally/local-evaluation-models/2026-11-08/official-assets.json').read_text())
    candidates = [a['href'] for page in old if page['name'] == 'object_detector' for a in page['assets']
                  if '/efficientdet_lite2/int8/' in a['href']]
    if len(candidates) != 1:
        raise ValueError('one official MediaPipe Lite2 int8 asset required')
    url = candidates[0].replace('/latest/', '/1/')
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(64*1024**2+1)
    if len(data) > 64*1024**2 or data[4:8] != b'TFL3':
        raise ValueError('bounded TFLite model required')
    path = args.output/'int8-efficientdet_lite2.tflite'
    path.write_bytes(data)
    models.append(dict(id='int8-efficientdet_lite2', path=str(path), url=url, bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(), source_url='https://developers.google.com/edge/mediapipe/solutions/vision/object_detector'))
    (args.output/'models.json').write_text(json.dumps(models, indent=2)+'\n')
    print(json.dumps(models[-1]), flush=True)


if __name__ == '__main__':
    main()
