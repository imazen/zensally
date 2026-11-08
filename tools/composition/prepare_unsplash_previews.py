#!/usr/bin/env python3
"""Create owned, bounded working previews; independent readers only verify PNGs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from PIL import Image

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'header_oracle', 'pipeline', 'png_reader', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    sources_path = args.cache/'sources.json'
    sources = json.loads(sources_path.read_text())
    headers = {r['id']: r for r in json.loads(args.header_oracle.read_text())['images']}
    if len(sources) != 144 or set(headers) != {r['id'] for r in sources}:
        raise ValueError('registered144-source cohort differs')
    if any(r['orientation'] != 1 or r['mode'] != 'RGB' or r['pixels'] > 200_000_000 for r in headers.values()):
        raise ValueError('registered source-header scope differs')
    protocol = dict(sources_sha256=sha(sources_path), header_oracle_sha256=sha(args.header_oracle),
        pipeline_sha256=sha(args.pipeline), png_reader_sha256=sha(args.png_reader),
        producer_sha256=sha(Path(__file__)), sources=144, long_edge=1280,
        source_max_pixels=200_000_000, encoded_input_limit=64*1024**2,
        working_payload_limit=2*1024**3, cms='strict-sRGB', resize='aspect-preserving Lanczos3, no upscaling',
        preprocessing_version='unsplash-owned-srgb1280-lanczos3-v1',
        trained=False, roles_assigned=False, canonical_holdouts_opened=False, paid_model_api_calls=0)
    if args.resume:
        if json.loads((args.output/'protocol.json').read_text()) != protocol:
            raise ValueError('resume protocol differs')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows, payload = [], 0
    for source in sources:
        identifier = source['provider_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', identifier):
            raise ValueError('unexpected provider ID')
        if (source['use'] != 'screening' or source['family_role'] is not None or source['training_eligible']
                or source['face_coverage_certified'] or source['human_rated']):
            raise ValueError('screening-only source contract differs')
        original = Path(source['file'])
        if original.stat().st_size != source['bytes'] or sha(original) != source['sha256']:
            raise ValueError('immutable source changed')
        header = headers[source['id']]
        record = args.output/(identifier+'.json')
        png, rgb = args.output/(identifier+'.png'), args.output/(identifier+'.rgb')
        if args.resume and record.exists():
            row = json.loads(record.read_text())
            if (row['source'] != source or sha(png) != row['png_sha256'] or sha(rgb) != row['rgb_sha256']
                    or not row['independent_png_rgb_exact']):
                raise ValueError('resume source/preview record differs')
        else:
            value = subprocess.run([str(args.pipeline), str(original), str(png), '1280'],
                capture_output=True, text=True)
            (args.output/(identifier+'.pipeline.log')).write_text(value.stdout+value.stderr)
            value.check_returncode()
            geometry = json.loads(value.stdout)
            if [geometry['source_width'], geometry['source_height']] != header['size']:
                raise ValueError('owned and independent source-header dimensions differ')
            reader = subprocess.run([str(args.png_reader), str(png), str(rgb)], capture_output=True, text=True)
            (args.output/(identifier+'.reader.log')).write_text(reader.stdout+reader.stderr)
            reader.check_returncode()
            with Image.open(png) as image:
                decoded = image.convert('RGB').tobytes()
                if (list(image.size) != [geometry['width'], geometry['height']]
                        or decoded != rgb.read_bytes() or image.getexif().get(274, 1) != 1):
                    raise ValueError('independent PNG sample/shape/orientation mismatch')
            row = dict(id=source['id'], source=source, native_size=header['size'], **geometry,
                png=str(png), rgb=str(rgb), png_sha256=sha(png), rgb_sha256=sha(rgb),
                independent_png_rgb_exact=True, preprocessing_version=protocol['preprocessing_version'],
                training_eligible=False, family_role=None, face_coverage_certified=False)
            with record.open('x') as stream:
                json.dump(row, stream, indent=2)
                stream.write('\n')
        rows.append(row)
        payload += png.stat().st_size+rgb.stat().st_size
        if payload > protocol['working_payload_limit']:
            raise ValueError('working preview2GiB payload budget exceeded; preserve completed files')
        (args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(), stage='owned-preview-readback',
            completed=len(rows), required=144, last_checked_unix=time.time()))+'\n')
        print(json.dumps(dict(id=source['id'], complete=len(rows), required=144)), flush=True)
    with (args.output/'manifest.json').open('x') as stream:
        json.dump(dict(protocol=protocol, rows=rows, payload_bytes=payload), stream, indent=2)
        stream.write('\n')
    (args.output/'progress.json').write_text(json.dumps(dict(pid=None, stage='complete', completed=144,
        required=144, last_checked_unix=time.time()))+'\n')
    print(json.dumps(dict(sources=144, payload_bytes=payload, independent_readbacks=144,
                         manifest_sha256=sha(args.output/'manifest.json'))))


if __name__ == '__main__':
    main()
