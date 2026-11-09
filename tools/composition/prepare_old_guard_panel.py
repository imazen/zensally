#!/usr/bin/env python3
"""Bind only the original25 DEV sources to unchanged local face/pose runners."""
import argparse
import json
from pathlib import Path
import torch
from artifact_identity import sha
from subject_native_faces import BINARY_SHA
from subject_tiled_faces import windows, extract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'face_manifest', 'binary', 'output']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    if sha(a.cache)!='2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031' or sha(a.binary)!=BINARY_SHA:
        raise ValueError('registered original cache/native detector differs')
    c = torch.load(a.cache, map_location='cpu', weights_only=False)
    sources = [s for s in c['sources'] if s['role']=='dev']
    if len(sources)!=25 or len({s['id'] for s in sources})!=25:
        raise ValueError('exact original25 DEV sources required')
    models = json.loads(a.face_manifest.read_text())['models']
    if {m['id'] for m in models}!={'float16-blaze_face_short_range', 'float16-blaze_face_full_range_sparse', 'float16-face_landmarker', 'float16-pose_landmarker_lite'}:
        raise ValueError('registered four local models required')
    for model in models:
        if sha(Path(model['path']))!=model['sha256']:
            raise ValueError('local teacher weights differ')
    images = []
    for source in sources:
        raw = Path(source['rgb'])
        w, h = source['working_size']
        if sha(raw)!=source['rgb_sha256'] or raw.stat().st_size!=w*h*3:
            raise ValueError('owned source pixels differ')
        images.append(dict(source, id=source['id'].replace(':','-'), source_id=source['id'],
                           width=w, height=h, benchmark=False))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'manifest.json').write_text(json.dumps(dict(images=images, models=models,
        source_cache_sha256=sha(a.cache), native_binary=str(a.binary), native_binary_sha256=BINARY_SHA,
        source_model_manifest_sha256=sha(a.face_manifest), api_calls=0, test_inference=False,
        source_roles_changed=False, diagnostic_maps_requested=False), indent=2)+'\n')
    (a.output/'inputs.tsv').write_text(''.join(f"{s['id']}\t{s['width']}\t{s['height']}\t{s['rgb']}\t0\n" for s in images))
    tiles = a.output/'tiles'
    tiles.mkdir()
    tile_rows = []
    for source in images:
        raw = Path(source['rgb']).read_bytes()
        for index, box in enumerate(windows(source['width'], source['height'])):
            path = tiles/(source['id']+f'-tile{index}.rgb')
            path.write_bytes(extract(raw, source['width'], source['height'], box))
            tile_rows.append(dict(id=path.stem, source_id=source['source_id'], source_box=box,
                source_size=[source['width'],source['height']], rgb=str(path), rgb_sha256=sha(path),
                width=box[2]-box[0], height=box[3]-box[1], benchmark=False))
    (tiles/'manifest.json').write_text(json.dumps(dict(images=tile_rows,models=[],api_calls=0,test_inference=False),indent=2)+'\n')
    (tiles/'inputs.tsv').write_text(''.join(f"{s['id']}\t{s['width']}\t{s['height']}\t{s['rgb']}\t0\n" for s in tile_rows))
    print(json.dumps(dict(dev_sources=len(images), tile_views=len(tile_rows), tile_bytes=sum(Path(s['rgb']).stat().st_size for s in tile_rows), api_calls=0, test_inference=False)))


if __name__=='__main__':
    main()
