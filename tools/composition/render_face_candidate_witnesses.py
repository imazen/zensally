#!/usr/bin/env python3
"""Bounded actual-pixel witnesses for local face candidate filtering failures."""
import argparse
import html
import json
from pathlib import Path
import sys
from artifact_identity import sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'crop-first'))
from rendered_pixels import render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['audit', 'native', 'resizer', 'encoder']:
        p.add_argument(name, type=Path)
    p.add_argument('--policy', default='native-yunet-tiled-interior/pad10/require1')
    a = p.parse_args()
    if sha(a.resizer)!='ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041' or sha(a.encoder)!='626afe7bbfbdafdf9b05d1b516b6f171caf19d50c80c68298fa07093a087da6d':
        raise ValueError('owned pixel executable differs')
    if sha(a.native/'SHA256SUMS.json')!='20adfd7777f535cce3ad328992b1f56ae76f7b0b87048f9a299bcdc98789441f' or sha(a.native/'manifest.json')!='c67aa540f626b985e61180067bce1b4c751797b6571b78f243dff3b3798fdc8b':
        raise ValueError('source pixel manifest differs')
    native = {r['source_id']: r for r in json.loads((a.native/'manifest.json').read_text())['images']}
    metadata = {r['id']: r['source'] for r in json.loads((a.audit/'sources.json').read_text())}
    rows = json.loads((a.audit/'predictions.json').read_text())
    policy = a.policy
    cases = [('openimages:9b15ccd240edae57', 1, 'previous group-head failure'),
             ('openimages:ff5b19334d988f53', 14, 'baseball head/body framing'),
             ('openimages:1b8020d5a8d620a5', 7, 'previous filter-induced head regression')]
    output = a.audit/'witnesses'
    output.mkdir(exist_ok=False)
    records = []
    articles = []
    for number, (sid, query, reason) in enumerate(cases):
        source = native[sid]
        raw_path = a.native/source['rgb']
        if sha(raw_path)!=source['rgb_sha256']:
            raise ValueError('source RGB differs')
        raw = raw_path.read_bytes()
        selected = {r['policy']: r for r in rows if r['id']==sid and r['query']==query
                    and r['model']=='replay-contrastive-seed31' and r['policy'] in ['raw', policy]}
        if set(selected)!={'raw', policy}:
            raise ValueError('required paired witness absent')
        ratio = selected['raw']['ratio']
        unit = 768//max(map(int, ratio.split(':')))
        panes = []
        figures = []
        for name, row in [('raw', selected['raw']), ('filtered', selected[policy]), ('contain', None)]:
            box = row['native_box'] if row else None
            rendered = render(raw, source['width'], source['height'], source['width']*3,
                              source['native_size'], box, ratio, output/f'case{number}-{name}',
                              a.resizer, a.encoder, unit=unit)
            panes.append(dict(name=name, record=row, **rendered))
            figures.append(f'<figure><img src="{Path(rendered["path"]).name}"><figcaption>{name}; weak cut={row["cut"] if row else "n/a"}</figcaption></figure>')
        records.append(dict(id=sid, query=query, reason=reason, source=metadata[sid], panes=panes))
        link = html.escape(metadata[sid]['source_url'], quote=True)
        articles.append(f'<article><h2>{html.escape(sid)} {ratio}: {html.escape(reason)}</h2><a href="{link}">Original source</a><div>{"".join(figures)}</div></article>')
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if total>32*1024*1024:
        raise ValueError('bounded witness storage budget exceeded')
    (output/'summary.json').write_text(json.dumps(dict(cases=records, policy=policy, asset_bytes=total,
        budget_bytes=32*1024*1024, api_calls=0, training_eligible=False,
        face_coverage_certified=False, model_promoted=False), indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Local face-filter failures</title><style>body{background:#222;color:white;font:16px system-ui;margin:24px}article div{display:flex;gap:12px}figure{margin:0;flex:1}img{width:100%;height:420px;object-fit:contain}</style><h1>Local face-filter witnesses</h1><p>Deliberately selected DEV failures and changes, seed31. Actual owned pixels; independent exact PNG readback. Incomplete weak head labels do not certify face or artistic quality. No training admission.</p>'+''.join(articles))
    print(json.dumps(dict(cases=len(cases), panes=sum(len(r['panes']) for r in records), asset_bytes=total)))


if __name__=='__main__':
    main()
