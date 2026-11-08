#!/usr/bin/env python3
"""Review native synthetic sources; optional center previews are not model crops."""
import argparse
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    jobs = json.loads((a.directory / 'plan.json').read_text())
    review = json.loads((a.directory / 'review.json').read_text())
    records = []
    for job in jobs:
        path = a.directory / (job['id'] + '.json')
        image = a.directory / (job['id'] + '.png')
        if not path.exists() or not image.exists():
            raise ValueError('gallery requires all planned native outputs')
        record = json.loads(path.read_text())
        if record['png_sha256'] != hashlib.sha256(image.read_bytes()).hexdigest():
            raise ValueError('generated source bytes changed')
        records.append(record)
    total = sum((Decimal(r['published_rate_usd']) for r in records), Decimal(0))
    config_path = a.directory / 'budget-config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    carry = Decimal(config.get('carry_usd', '0'))
    ceiling = Decimal(config.get('ceiling_usd', '1'))
    if not Decimal(0) < ceiling <= Decimal('2.50') or total + carry >= ceiling:
        raise ValueError('experiment exceeded owner budget')
    parts = []
    esc = html.escape
    for index, record in enumerate(records):
        identity = record['id']
        w, h = map(int, record['size'].split('x'))
        intent = (f"Requested background: {record['background_requested']} · "
                  f"requested brand: {record['brand_requested'] or 'none'}") if 'brand_requested' in record else 'Initial unbranded ecommerce exploration'
        parts.append(f'''<section id="{esc(identity)}" data-width="{w}" data-height="{h}">
<h2>{index + 1}. {esc(identity.replace('-', ' '))}</h2>
<figure><img src="{esc(identity)}.png" width="{w}" height="{h}" loading="lazy" alt="Synthetic {esc(record['product'])} commercial photograph"></figure>
<p>{esc(record['size'])} · high quality / high complexity · ${esc(record['published_rate_usd'])} including prompt tokens</p>
<p><strong>Crop challenge:</strong> {esc(record['crop_challenge'])}</p>
<p><strong>Coordinator inspection:</strong> {esc(review['notes'][identity])}</p>
<p>{esc(intent)}</p>
<p>Requested people: {record['people_requested']} · requested animals: {record['animals_requested']} · scenario family: {esc(record['scenario_family'])}. Requested counts are not verified labels.</p>
<p><a href="{esc(identity)}.png">Open native PNG</a> · <a href="{esc(identity)}.json">Prompt, settings and native usage</a></p>
<details><summary>Full generation prompt</summary><pre>{esc(record['prompt'])}</pre></details></section>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sunburst synthetic photo exploration</title><style>
:root{color-scheme:dark;--crop-ratio:1}*{box-sizing:border-box}body{font:16px/1.55 system-ui;margin:24px;background:#12171c;color:#e8eef5}main{max-width:1450px;margin:auto}h1{line-height:1.15}p{max-width:1000px}a{color:#8ed2ff}header{padding-bottom:24px;border-bottom:1px solid #566575}.controls{position:sticky;top:0;z-index:2;background:#18232fee;padding:12px;display:flex;flex-wrap:wrap;gap:16px}select{font:inherit;padding:5px;background:#263647;color:inherit}#cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}section{min-width:0;padding:16px;background:#1a242f;border:1px solid #405165;border-radius:12px}h2{font-size:20px}figure{margin:0;width:100%;background:#080d12;display:grid;place-items:center}img{display:block;width:100%;height:auto;max-height:680px;object-fit:contain}.preview figure{aspect-ratio:var(--crop-ratio);display:block;overflow:hidden}.preview figure img{width:100%;height:100%;max-height:none;object-fit:cover;object-position:50% 50%}pre{white-space:pre-wrap;font:14px/1.5 ui-monospace,monospace}details{border-top:1px solid #405165;padding-top:8px}#warning{color:#ffe4a3}.price{font-size:22px;font-weight:650}@media(max-width:800px){#cards{grid-template-columns:1fr}body{margin:12px}}
</style><main><header><h1>Synthetic source photographs for crop research</h1>
<p>GPT-Image-2.6 Sunburst · high quality · high complexity. Fictional people/products, generated for crop research. These are synthetic exploration fixtures with no training admission, face-coverage certification or human crop judgments.</p>
<p class="price">TOTAL_COST_PLACEHOLDER</p>
<p>All source PNGs retain their native aspect ratio. The optional centered previews deliberately demonstrate a simple baseline: they are not outputs of the crop generator and can remove or cut faces. Use “Native source” to inspect the complete photograph.</p>
<p><a href="calculator.json">Exact-ratio calculator estimates</a> · <a href="plan.json">All authored prompts and intended challenges</a> · <a href="review.json">Visual inspection notes</a></p></header>
<div class="controls"><label>View <select id="aspect"><option value="source">Native source</option><option value="1">1:1 center preview</option><option value="1.3333333333333333">4:3 center preview</option><option value="0.75">3:4 center preview</option><option value="2">2:1 center preview</option><option value="0.5">1:2 center preview</option><option value="3">3:1 center preview</option><option value="0.3333333333333333">1:3 center preview</option></select></label><span id="warning">Native source photographs</span></div>
<div id="cards">CARDS_PLACEHOLDER</div></main><script>
const aspect=document.querySelector('#aspect'),cards=document.querySelector('#cards'),warning=document.querySelector('#warning');
aspect.addEventListener('change',()=>{const source=aspect.value==='source';cards.classList.toggle('preview',!source);document.documentElement.style.setProperty('--crop-ratio',source?'1':aspect.value);warning.textContent=source?'Native source photographs':'Centered baseline preview only; faces/products may be lost. Not a model recommendation.'});
</script></html>'''
    page = page.replace('TOTAL_COST_PLACEHOLDER', f'{len(records)} images · ${total} for this set · ${total + carry} including prior usage · published rates including prompt tokens, not invoice-verified')
    page = page.replace('CARDS_PLACEHOLDER', '\n'.join(parts))
    target = a.directory / 'index.html'
    if target.exists():
        target.rename(target.with_name(target.name + '.bak-' + str(time.time_ns())))
    target.write_text(page)
    print(json.dumps({'images': len(records), 'published_rate_total_usd': str(total), 'gallery': str(target)}))


if __name__ == '__main__':
    main()
