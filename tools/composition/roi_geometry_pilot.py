#!/usr/bin/env python3
"""Bounded cached-detection ROI crop pilot; no training, paid calls or TEST."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import html
import json
from math import ceil, floor
from pathlib import Path
import shutil

from artifact_identity import sha
from roi_crop_geometry import solve, rank_regions, envelope
from subject_policy import face_relation
from subject_tiled_faces import iou


RATIOS = ['1:1', '4:3', '3:4', '16:9', '9:16', '2:1', '1:2', '3:1', '1:3']


def proposals(source, group, detections):
    w, h = source['native_size']
    pw, ph = source['working_size']
    regions = []
    for index, detection in enumerate(detections):
        category = max(detection['categories'], key=lambda c: c['score'])
        if category['score'] < .3:
            continue
        x, y, cw, ch = detection['xywh']
        box = [max(0, floor(Fraction(x*w, pw))), max(0, floor(Fraction(y*h, ph))),
               min(w, ceil(Fraction((x+cw)*w, pw))), min(h, ceil(Fraction((y+ch)*h, ph)))]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('empty detected object')
        regions.append(dict(box=box, confidence=category['score'], kind=category['name'],
                            extent='whole-detected-object', provenance=f'Lite2/detection{index}'))
    dedup = []
    for row in rank_regions(regions, w, h):
        if all(row['kind'] != r['kind'] or iou(row['box'], r['box']) <= .7 for r in dedup):
            dedup.append(row)
    # A group is an alternative, never a constraint requiring all people.
    people = [r for r in dedup if r['kind'] == 'person']
    if len(people) > 1:
        dedup.append(dict(box=envelope([r['box'] for r in people]),
                          confidence=min(r['confidence'] for r in people), kind='person-group',
                          extent='whole-detected-group', provenance='union of detected people'))
    ranked = rank_regions(dedup, w, h)
    # Face/pose context is explicitly a portrait-detail fallback. It is not
    # advertised as a whole person/product crop. No upright upper-body guess.
    faces = sorted(set(map(tuple, group['faces']+group['pose'])))
    portraits = []
    for i, f in enumerate(faces):
        dx, dy = (f[2]-f[0]+1)//2, (f[3]-f[1]+1)//2
        box = [max(0, f[0]-dx), max(0, f[1]-dy), min(w, f[2]+dx), min(h, f[3]+dy)]
        portraits.append(dict(box=box, confidence=1., kind='portrait-detail', extent='head-context-only',
                              provenance=f'cached face/pose box{i}; confidence not calibrated'))
    return ranked + rank_regions(portraits, w, h), [list(f) for f in faces]


def svg(path, box, shape, label):
    w, h = shape
    x, y, right, bottom = box
    return (f'<figure><svg viewBox="{x} {y} {right-x} {bottom-y}" style="overflow:hidden">'
            f'<image href="{path}" width="{w}" height="{h}"/></svg>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('objects', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    packet = json.loads((args.parent/'SHA256SUMS.json').read_text())
    for relative in ['regions.json', 'all-risk/predictions.json']:
        if sha(args.parent/relative) != packet[relative]:
            raise ValueError('frozen parent changed')
    regions = json.loads((args.parent/'regions.json').read_text())
    if sha(args.objects) != regions['inputs'][str(args.objects)]:
        raise ValueError('cached detections differ')
    detections = {r['id']: r['result']['detections'] for r in json.loads(args.objects.read_text())['rows']}
    previous = {(r['id'], r['ratio'], r['query'] % 2): r['native'] for r in
                json.loads((args.parent/'all-risk/predictions.json').read_text())
                if r['model'] == 'unfiltered' and r['applicability'] == 'all'}
    args.output.mkdir(parents=True, exist_ok=False)
    records, articles = [], []
    for si, (source, group) in enumerate(zip(regions['sources'], regions['groups'])):
        if source['role'] not in ['dev', 'development']:
            raise ValueError('registered development only')
        w, h = source['native_size']
        pw, ph = source['working_size']
        if sha(Path(source['png'])) != source['png_sha256']:
            raise ValueError('owned source preview changed')
        name = f'source{si:02d}.png'
        shutil.copyfile(source['png'], args.output/name)
        candidates, faces = proposals(source, group, detections[source['id']])
        for ratio in RATIOS:
            p, q = map(int, ratio.split(':'))
            k = min(w//p, h//q)
            x, y = (w-k*p)//2, (h-k*q)//2
            center = [x, y, x+k*p, y+k*q]
            options = []
            infeasible = 0
            for rank, candidate in enumerate(candidates):
                result = solve(w, h, ratio, candidate['box'], faces)
                if result['outer'] is None:
                    infeasible += 1
                    continue
                if any(result['outer'] == v['outer'] and result['inner'] == v['inner'] for v in options):
                    continue
                result.update(subject=candidate, rank=rank)
                for crop in [result['outer'], result['inner']]:
                    if crop is not None and any(face_relation(crop, f) == 'cut' for f in faces):
                        raise ValueError('supplied-face cut')
                options.append(result)
                if len(options) == 3:
                    break
            selected = options[0] if options else None
            row = dict(id=source['id'], family_id=source['family_id'], role=source['role'], ratio=ratio,
                       source_size=[w, h], center=center, options=options,
                       unavailable=selected is None, candidate_count=len(candidates), infeasible_examined=infeasible,
                       supplied_faces=faces, center_face_cuts=sum(face_relation(center, f) == 'cut' for f in faces),
                       selected_face_cuts=0 if selected else None,
                       whole_object=bool(selected and selected['subject']['extent'] != 'head-context-only'))
            records.append(row)
            def pane(box, label):
                # Native coordinates project directly to the existing owned
                # preview. SVG viewport clips, rather than just moving overlays.
                scaled = [box[0]*pw/w, box[1]*ph/h, box[2]*pw/w, box[3]*ph/h]
                return svg(name, scaled, (pw, ph), label)
            panes = pane([0, 0, w, h], 'Source') + pane(center, 'Plain center crop')
            panes += pane(previous[source['id'], ratio, 0], 'Previous learned Minimal')
            if selected:
                panes += pane(selected['outer'], f"ROI Minimal: {selected['subject']['kind']}")
                if selected['inner']:
                    panes += pane(selected['inner'], 'ROI Maximal: same subject and retained faces')
                detail = html.escape(json.dumps(selected['subject']))
            else:
                detail = 'No supplied subject fits without intersecting supplied faces: contain/pad.'
            articles.append(f'<article><h2>{html.escape(source["id"])} {ratio}</h2><p>{detail}</p>'
                            f'<div class="panes">{panes}</div><details><summary>All ranked options and coordinates</summary>'
                            f'<pre>{html.escape(json.dumps(row, indent=2))}</pre></details></article>')
    counts = Counter('contain' if r['unavailable'] else 'whole-object' if r['whole_object'] else 'portrait-detail' for r in records)
    summary = dict(sources=len(regions['sources']), requests=len(records), outcomes=dict(counts),
                   nested_pairs=sum(bool(r['options'] and r['options'][0]['inner']) for r in records),
                   center_requests_cutting_supplied_faces=sum(r['center_face_cuts'] > 0 for r in records),
                   selected_requests_cutting_supplied_faces=sum((r['selected_face_cuts'] or 0) > 0 for r in records),
                   offered=sum(not r['unavailable'] for r in records),
                   api_calls=0, model_training=False, test_inference=False, artistic_quality_verified=False,
                   face_coverage_certified=False, source_family_roles_changed=False,
                   inputs={str(args.parent/'SHA256SUMS.json'): sha(args.parent/'SHA256SUMS.json'), str(args.objects): sha(args.objects)},
                   limitation='Geometry protects supplied boxes only. Ranking is confidence/area heuristic. No gaze, sharpness, text/logo detector or semantic subject-choice model.')
    for file, data in [('predictions.json', records), ('summary.json', summary)]:
        (args.output/file).write_text(json.dumps(data, indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ranked ROI geometry pilot</title>'
        '<style>body{font:16px system-ui;background:#222;color:#eee;margin:20px}.panes{display:flex;gap:12px;flex-wrap:wrap}'
        'figure{margin:0;flex:1;min-width:150px;max-width:360px}svg{width:100%;height:270px}pre{white-space:pre-wrap}</style>'
        '<h1>Ranked ROI geometry: development pilot</h1><p>Whole detected objects first; portrait-detail fallback is explicit. '
        'No guarantee for missed faces or artistic composition. Source, center crop and earlier learned crop are controls.</p>'
        + ''.join(articles))
    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 64*1024**2:
        raise ValueError('64MiB pilot storage budget exceeded')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
