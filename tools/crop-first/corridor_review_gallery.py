#!/usr/bin/env python3
"""Show actual interval pixels and joined blinded judgments, preserving source links."""
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image
from direct_zones import sha

HTML='''<!doctype html><meta charset="utf-8"><title>Nested Minimal / Maximal crop review</title>
<style>body{font:16px system-ui;background:#191c22;color:#eee;margin:24px}p{max-width:1120px;line-height:1.45}a{color:#9dd6ff}article{border-top:2px solid #697182;margin-top:24px;padding-top:16px}.source{width:100%;max-width:560px;height:390px;object-fit:contain}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0;background:#292e38;padding:12px}.frame{width:100%;height:240px;object-fit:contain}figcaption{font-size:14px}h2{font-size:20px}select{max-width:100%;font:inherit}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>Nested Minimal / Maximal crop review</h1><p>Outer is the maximum-area Minimal crop. Inner is the tighter Maximal crop at exactly the same aspect. Three intermediate frames and a full-source containment control use actual owned-codec PNGs. Teacher proposals and model judgments are weak annotations, not human artistic gold or complete face coverage. Every displayed interval passes geometry against its supplied face boxes; source detection completeness remains unproven.</p><p id="status"></p><select id="selection"></select><main></main>
<script>const data=DATA;const selection=document.querySelector('#selection');selection.add(new Option('All intervals','all'));for(const g of data.groups)selection.add(new Option(g.id+' · '+g.ratio+' · '+g.role,g.id));
function element(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
function render(){document.querySelector('#status').textContent=data.groups.length+' intervals · '+data.presentations+' completed blinded reviews · $'+data.settled_usd+' settled in the new labeling pilot. Labels reflect the saved analysis snapshot.';const main=document.querySelector('main');main.replaceChildren();for(const g of data.groups){if(selection.value!=='all'&&selection.value!==g.id)continue;const a=element('article');a.dataset.id=g.id;a.append(element('h2',g.id+' · '+g.ratio+' · '+g.role+' · annotation stratum '+g.cell));const link=element('a',g.title+' — '+g.author);link.href=g.source_url;a.append(link);const src=element('img');src.className='source';src.src=g.image;src.alt='Original full photograph';a.append(src);a.append(element('p',g.integer_crops+' exact-aspect integer crops in this interval. Endpoint excellence confirmed by all three reviews: '+g.excellent+'. The whole interval has no training admission in this viewer.'));const grid=element('div');grid.className='grid';for(const c of g.candidates){const f=element('figure');f.dataset.origin=c.origin;const im=element('img');im.className='frame';im.src=c.image;im.alt=c.origin;f.append(im);f.append(element('h3',c.origin==='inner'?'Maximal inner':c.origin==='outer'?'Minimal outer':c.origin));for(const label of c.judgments)f.append(element('figcaption',label));grid.append(f)}a.append(grid);main.append(a)}}selection.onchange=render;render();</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    summary=json.loads((a.analysis/'summary.json').read_text());reviews=json.loads((a.analysis/'reviews.json').read_text());source={r['id']:r for r in json.loads((a.root/'plan.json').read_text())['cases']};groups={}
    for path in a.root.rglob('plan.json'):
        for group in json.loads(path.read_text()).get('groups',[]):
            if group['pair'].get('pair_protocol',1)==2 and group['pair']['ready_for_rendered_review']:groups[group['case_id']]=group
    a.output.mkdir(exist_ok=False);(a.output/'assets').mkdir();copied={};total=0
    def copy(path,expected):
        nonlocal total
        raw=Path(path).read_bytes();digest=sha(raw)
        if digest!=expected:raise ValueError('gallery pixel identity differs')
        if digest not in copied:
            target=a.output/'assets'/(digest+'.png');target.write_bytes(raw);total+=len(raw);copied[digest]='assets/'+target.name
        if total>128*1024**2:raise ValueError('128MiB gallery asset budget exceeded')
        return copied[digest]
    rows=[]
    for identity,group in sorted(groups.items()):
        case=source[group['source_id']];info=case['source_information']['source'];image=copy(case['image']['path'],case['image']['sha256'])
        with Image.open(case['image']['path']) as original:source_size=list(original.size)
        candidates=[];joined=[r for r in reviews if r['group']==identity]
        for candidate in group['candidates']:
            origin=candidate['origin'];labels=[]
            for row in joined:
                grade=row['grades'][origin]
                if grade['png_sha256']!=candidate['sha256'] or grade['pixel_sha256']!=candidate['rgb_sha256']:raise ValueError('rendered judgment association differs')
                labels.append(row['model']+' / '+row['presentation']+': '+grade['composition_quality']+'; acceptable '+grade['acceptable']+'; new face cut '+grade['new_face_truncation'])
            candidates.append(dict(origin=origin,image=copy(candidate['path'],candidate['sha256']),width=candidate['width'],height=candidate['height'],native_box=candidate['native_box'],judgments=labels or ['Not yet judged in this snapshot.']))
        interval=next((r for r in summary['intervals'] if r['group']==identity),None)
        rows.append(dict(id=identity,source_id=group['source_id'],role=case['role'],cell=case['cell'],ratio=group['ratio'],native_size=[case['width'],case['height']],source_size=source_size,
            image=image,source_url=info['OriginalLandingURL'],author=info['Author'],title=info['Title'],source_information=case['source_information'],
            integer_crops=group['pair']['integer_crop_count'],pair=group['pair'],excellent=bool(interval and interval['complete_three_presentations'] and interval['all_reviewed_endpoints_excellent']),candidates=candidates))
    data=dict(groups=rows,presentations=len(reviews),settled_usd=summary['settled_ledger_usd'],asset_bytes=total,screenshot_budget_bytes=16*1024**2,model_promoted=False,training_started=False,
        analysis_summary_sha256=sha((a.analysis/'summary.json').read_bytes()))
    (a.output/'data.json').write_text(json.dumps(data,indent=2)+'\n');(a.output/'index.html').write_text(HTML.replace('DATA',json.dumps(data).replace('</','<\\/')))
    print(json.dumps(dict(groups=len(rows),assets=len(copied),asset_bytes=total)))


if __name__=='__main__':main()
