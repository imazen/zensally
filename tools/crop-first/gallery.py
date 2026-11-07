#!/usr/bin/env python3
"""Static, pixel-linked review sheet for unadmitted pilot preferences."""
import argparse,csv,html,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pilot',type=Path);a=p.parse_args()
    result=json.loads((a.pilot/'cli-labels.json').read_text());esc=html.escape;parts=[]
    for r in result['records']:
        sid=r['source_id'];mode=r['task_id'].split('-',1)[1]
        cs=[c for c in csv.DictReader((a.pilot/sid/'candidates.tsv').open(),delimiter='\t') if c['mode']==mode]
        c=cs[0];sw,sh=c['source_w'],c['source_h']
        overlay=f'<svg viewBox="0 0 {sw} {sh}" role="img" aria-label="Original with candidate boxes"><image href="{sid}/source.png" width="{sw}" height="{sh}"/>'
        for row,color in zip(cs,['#ef4444','#22c55e']):
            overlay+=f'<rect x="{row["x"]}" y="{row["y"]}" width="{row["w"]}" height="{row["h"]}" fill="none" stroke="{color}" stroke-width="{int(sw)/150}"/>'
        overlay+='</svg>'
        label=r['label'];parts.append(f'<section><h2>{esc(r["task_id"])} · {c["ratio_w"]}:{c["ratio_h"]}</h2><div class="images"><figure>{overlay}<figcaption>Original · A red / B green</figcaption></figure>')
        for row in cs:
            letter=row['candidate'];parts.append(f'<figure><img src="{sid}/{row["file"]}" alt="Candidate {letter}"><figcaption>{letter} · source xywh ({row["x"]}, {row["y"]}, {row["w"]}, {row["h"]})</figcaption></figure>')
        parts.append(f'</div><p><b>Teacher preference: {esc(label["preferred"])}</b>. {esc(label["reason"])}</p><p>A acceptable: {label["a_acceptable"]}; B acceptable: {label["b_acceptable"]}. Face coverage unverified.</p></section>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Crop preference annotation pilot</title><style>body{font:16px system-ui;margin:24px;background:#14171c;color:#eee}section{border-top:1px solid #596170;margin-top:32px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}figure{margin:0}img,svg{display:block;max-width:100%;width:100%;height:420px;object-fit:contain}figcaption{font-size:14px;margin-top:8px}p{max-width:1000px}a{color:#9df}@media(max-width:700px){.images{grid-template-columns:1fr}}</style><h1>Own-source crop preference pilot</h1><p>Two TRAIN sources, four comparisons. These deliberately unfiltered annotation examples may contain bad crops. They are never deployment recommendations. Terra judgments are unreviewed weak labels; no training admission, human calibration or face-safety certification.</p>'''+''.join(parts)+'<p><a href="cli-labels.json">Judgments and provenance</a> · <a href="protocol.json">Frozen protocol</a></p>'
    (a.pilot/'index.html').write_text(page)


if __name__=='__main__':main()
