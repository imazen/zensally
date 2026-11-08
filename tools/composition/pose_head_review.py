#!/usr/bin/env python3
"""Render fixed before/after head-bound witnesses with owned crop pixels."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['screen', 'acquired', 'experiment', 'resizer', 'encoder', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.resizer) != 'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041':
        raise ValueError('pinned resizer differs')
    if sha(args.encoder) != '626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':
        raise ValueError('pinned PNG encoder differs')
    sources = json.loads((args.acquired/'sources.json').read_text())
    comparisons, counts = [], dict(requests=0, old_crop=0, new_crop=0, changed=0, became_contain=0)
    args.output.mkdir(parents=True, exist_ok=False)
    gallery, rendered = [], []
    for source_id in ['11048', '11214']:
        source = next(s for s in sources if s['id'] == source_id)
        original = json.loads((args.screen/'crop-proposals'/(source_id+'.json')).read_text())
        revised = json.loads((args.experiment/'proposals'/(source_id+'.json')).read_text())
        if not all(h in revised['protected_head_pixels'] for h in original['protected_head_pixels']):
            raise ValueError('an original face constraint was removed')
        preview = args.screen/'gallery-assets'/(source_id+'.png')
        source_name = source_id+'-source.png'
        shutil.copy2(preview, args.output/source_name)
        jobs = []
        for before, after in zip(original['tasks'], revised['tasks'], strict=True):
            if before['ratio'] != after['ratio']:
                raise ValueError('task ratio differs')
            p, q = map(int, before['ratio'].split(':'))
            for bs, ass in zip(before['selected'], after['selected'], strict=True):
                boxes = []
                for task, selected, is_new in [(before, bs, False), (after, ass, True)]:
                    index = selected['detected_head_guarded_candidate']
                    box = None if index is None else task['candidates'][index]['pixel_box']
                    boxes.append(box)
                    if box:
                        l, t, r, b = box
                        if not (0 <= l < r <= source['width'] and 0 <= t < b <= source['height']
                                and (r-l)*q == (b-t)*p):
                            raise ValueError('integer crop contract failed')
                        if is_new and any(not(l <= h[0] and t <= h[1] and r >= h[2] and b >= h[3])
                                          for h in revised['protected_head_pixels']):
                            raise ValueError('new crop clips a supplied region')
                old, new = boxes
                counts['requests'] += 1
                counts['old_crop'] += int(old is not None)
                counts['new_crop'] += int(new is not None)
                counts['changed'] += int(old != new)
                counts['became_contain'] += int(old is not None and new is None)
                comparison = dict(id=source_id, ratio=before['ratio'], mode=bs['mode'], before=old, after=new)
                comparisons.append(comparison)
                if before['ratio'] not in ['3:1', '1:3']:
                    continue
                row = dict(comparison, source_image=source_name, images=[])
                for version, box in zip(['before', 'after'], boxes):
                    if box is None:
                        row['images'].append(dict(file=source_name, action='contain', version=version))
                    else:
                        l, t, r, b = box
                        units = min(512//p, 512//q, (r-l)//p)
                        ow, oh = p*units, q*units
                        name = f'{source_id}-{p}x{q}-{bs["mode"]}-{version}'
                        jobs.append(dict(file=name, pixel_box=box, width=ow, height=oh))
                        row['images'].append(dict(file=name+'.png', action='crop', version=version))
                gallery.append(row)
        native = Path(source['directory'])/'source.rgb'
        if sha(native) != source['rgb_sha256']:
            raise ValueError('native RGB differs')
        specs = ['{},{},{},{},{},{}'.format(j['pixel_box'][0], j['pixel_box'][1],
            j['pixel_box'][2]-j['pixel_box'][0], j['pixel_box'][3]-j['pixel_box'][1], j['width'], j['height'])
            for j in jobs]
        if jobs:
            with native.open('rb') as stream:
                pixels = subprocess.check_output([str(args.resizer), str(source['width']), str(source['height']),
                    str(source['width']*3), ';'.join(specs)], stdin=stream)
            if len(pixels) != sum(j['width']*j['height']*3 for j in jobs):
                raise ValueError('render output length differs')
            offset = 0
            for job in jobs:
                size = job['width']*job['height']*3
                raw, png = args.output/(job['file']+'.rgb'), args.output/(job['file']+'.png')
                raw.write_bytes(pixels[offset:offset+size])
                subprocess.run([str(args.encoder), '--from-rgb', str(job['width']), str(job['height']),
                                str(raw), str(png)], check=True)
                rendered.append(dict(job, rgb=str(raw), png=str(png), rgb_sha256=sha(raw), png_sha256=sha(png)))
                offset += size
    doc = dict(summary=counts, rows=comparisons, rendered=rendered, gallery=gallery,
        face_coverage_certified=False, training_eligible=False, quality_model_promoted=False)
    (args.output/'comparison.json').write_text(json.dumps(doc, indent=2)+'\n')
    template = Path(__file__).with_name('pose_head_review.html').read_text()
    (args.output/'index.html').write_text(template.replace('/*DOCUMENT*/null', json.dumps(gallery)))
    print(json.dumps(dict(counts, rendered_crops=len(rendered), reviewed_tasks=len(gallery))))


if __name__ == '__main__':
    main()
