#!/usr/bin/env python3
"""Bind fixed local face/pose models to the existing real72 owned RGB inputs."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha

MODELS={
    'float16-blaze_face_short_range':'b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f',
    'float16-blaze_face_full_range_sparse':'2c3728e6da56f21e21a320433396fb06d40d9088f2247c05e5635a688d45dfe1',
    'float16-pose_landmarker_lite':'59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
    'float16-face_landmarker':'64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['tasks','models','output']:p.add_argument(name,type=Path)
    a=p.parse_args();tasks=[json.loads(Path(path).read_text()) for path in json.loads(a.tasks.read_text())]
    if len(tasks)!=72 or len({t['family_id'] for t in tasks})!=72 or sum(t['role']=='fit' for t in tasks)!=48 or sum(t['role']=='dev' for t in tasks)!=24:raise ValueError('fixed72family48/24 panel required')
    acquired=Path(tasks[0]['image']['path']).parents[1]
    if sha(acquired/'SHA256SUMS.json')!='93f63d69180d82f197e34a733005118de260eda728e18998ced3194959581bb1':raise ValueError('frozen owned acquisition differs')
    source={r['id']:r for r in json.loads((acquired/'manifest.json').read_text())['rows']};images=[]
    for task in tasks:
        row=source[task['id']];r=row['source'];path=Path(task['image']['path']);rgb=path.with_suffix('.rgb');g=row['geometry']
        if (task['role'],task['family_id'])!=(r['role'],r['family_id']) or sha(path)!=task['image']['sha256'] or sha(rgb)!=row['preview.rgb_sha256'] or rgb.stat().st_size!=g['width']*g['height']*3:raise ValueError('fixed owned source/role/pixels differ')
        images.append(dict(id=task['id'],family_id=task['family_id'],role=task['role'],width=g['width'],height=g['height'],
            native_size=[task['width'],task['height']],rgb=str(rgb),rgb_sha256=row['preview.rgb_sha256'],display=str(path),benchmark=False))
    models=[]
    for record in json.loads((a.models/'inventory.json').read_text()):
        name=Path(record['file']).stem
        if name not in MODELS:continue
        path=a.models/record['file']
        if sha(path)!=MODELS[name] or record['sha256']!=MODELS[name]:raise ValueError('fixed local cue weights differ')
        models.append(dict(id=name,path=str(path),sha256=MODELS[name],bytes=path.stat().st_size))
    if len(models)!=4:raise ValueError('all four registered face/pose models required')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(images=images,models=models,source_tasks_sha256=sha(a.tasks),source_acquisition_sha256=sha(acquired/'SHA256SUMS.json'),
        purpose='Local face/pose cue suitability for choosing a person and avoiding partial faces; no new crop labels',
        face_confidence=.5,pose_count=8,face_landmark_count=16,synthetic_sources=False,face_coverage_certified=False,
        source_roles_changed=False,test_inference=False,paid_api_calls=0,storage_allowance_bytes=128*1024**2)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(sources=72,models=4,expected_calls=288)))


if __name__=='__main__':main()
