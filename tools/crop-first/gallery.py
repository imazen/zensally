#!/usr/bin/env python3
"""Pixel-linked review sheet for unadmitted pair/pool preferences."""
import argparse,csv,html,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pilot',type=Path);p.add_argument('--expanded',action='store_true');p.add_argument('--labels',default='cli-labels.json');p.add_argument('--output',default='index.html');a=p.parse_args()
    result=json.loads((a.pilot/a.labels).read_text());esc=html.escape;parts=[]
    for r in result['records']:
        sid=r['source_id'];mode=r['task_id'].split('-',1)[1];label=r['label'];prefix=f'crops/{sid}' if a.expanded else sid
        cs=[c for c in csv.DictReader((a.pilot/prefix/'candidates.tsv').open(),delimiter='\t') if c['task_id']==r['task_id']]
        c=cs[0];sw,sh=c['source_w'],c['source_h'];preferred=label['preferred']
        winner=next((c for c in cs if c['candidate']==preferred),None)
        overlay=f'<svg viewBox="0 0 {sw} {sh}" role="img" aria-label="Original with selected crop"><image href="{prefix}/source.png" width="{sw}" height="{sh}"/>'
        for row,color,dash in [(cs[0],'#aaa','24 16')]+([(winner,'#22c55e','none')] if winner else []):
            overlay+=f'<rect x="{row["x"]}" y="{row["y"]}" width="{row["w"]}" height="{row["h"]}" fill="none" stroke="{color}" stroke-dasharray="{dash}" stroke-width="{int(sw)/170}"/>'
        overlay+='</svg>'
        parts.append(f'<section data-selected="{str(winner is not None).lower()}" data-source="{sid}" data-aspect="{c["ratio_w"]}:{c["ratio_h"]}" data-mode="{c["mode"]}"><h2>{esc(r["task_id"])} · {c["ratio_w"]}:{c["ratio_h"]}</h2><div class="hero"><figure>{overlay}<figcaption>Original · selected crop green / first baseline dashed gray</figcaption></figure>')
        if winner:parts.append(f'<figure><img loading="lazy" src="{prefix}/{winner["file"]}" alt="Selected crop {preferred}"><figcaption>Teacher selection: {esc(preferred)}</figcaption></figure>')
        if not winner:parts.append(f'<figure><img loading="lazy" src="{prefix}/{cs[0]["file"]}" alt="Rejected center baseline"><figcaption>Contain/pad · no crop selected. Center baseline shown for audit.</figcaption></figure>')
        parts.append(f'</div><p>{esc(label["reason"])}</p><details><summary>All {len(cs)} rendered proposals</summary><div class="images">')
        for row in cs:
            name=row['candidate'];origin='Terra proposal' if 'vision-proposal' in row.get('generators','') else 'Baseline'
            accepted=name in label['acceptable'] if 'acceptable' in label else label.get(name.lower()+'_acceptable',False)
            parts.append(f'<figure class="{"selected" if name==preferred else ""}"><img loading="lazy" src="{prefix}/{row["file"]}" alt="Candidate {esc(name)}"><figcaption>{esc(name)} · {origin} · teacher {"accepts" if accepted else "rejects"}<br>source xywh ({row["x"]}, {row["y"]}, {row["w"]}, {row["h"]})</figcaption></figure>')
        parts.append('</div></details></section>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Crop composition proposals</title><style>body{font:16px system-ui;margin:24px;background:#14171c;color:#eee}section{border-top:1px solid #596170;margin-top:32px}.hero,.images{display:grid;gap:20px}.hero{grid-template-columns:repeat(2,minmax(0,1fr))}.images{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:20px}figure{margin:0;padding:6px}img,svg{display:block;max-width:100%;width:100%;height:400px;object-fit:contain}.images img{height:300px}.selected{outline:2px solid #22c55e}figcaption{font-size:14px;margin-top:8px}p{max-width:1000px}summary{cursor:pointer;padding:12px 0}a{color:#9df}@media(max-width:900px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.images,.hero{grid-template-columns:1fr}}</style><h1>Crop composition proposals</h1><div id="controls"><label>Source <select id="source"></select></label> <label>Aspect <select id="aspect"></select></label> <label>Mode <select id="mode"></select></label> <button id="previous">Previous</button> <button id="next">Next</button> <button id="accepted">Next selected crop</button> <span id="count"></span></div><p>Compare center/interior baselines and directly proposed Terra compositions. Minimal uses maximum-area framing at the requested aspect. Maximal includes broad context and tighter alternatives. Selected crops are unreviewed teacher preferences, with no training admission, human calibration or face-safety certification.</p>'''+''.join(parts)+'<p><a href="cli-labels.json">Judgments and provenance</a> · <a href="protocol.json">Frozen protocol</a></p>'
    page+="""<script>
const sections=[...document.querySelectorAll('section')];const params=new URLSearchParams(location.search);
for(const key of ['source','aspect','mode']){const select=document.getElementById(key);for(const value of [...new Set(sections.map(s=>s.dataset[key]))]){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);}const desired=params.get(key==='source'?'id':key)||(key==='aspect'?'1:8':null);if(desired&&[...select.options].some(o=>o.value===desired))select.value=desired;select.addEventListener('change',render);}
function render(){let visible=0;for(const s of sections){s.hidden=['source','aspect','mode'].some(k=>s.dataset[k]!==document.getElementById(k).value);if(!s.hidden)visible++;}document.getElementById('count').textContent=sections.length+' judged requests; '+visible+' shown';const q=new URLSearchParams({id:document.getElementById('source').value,aspect:document.getElementById('aspect').value,mode:document.getElementById('mode').value});history.replaceState(null,'','?'+q);}
function move(step,accepted=false){let i=sections.findIndex(s=>!s.hidden);for(let n=0;n<sections.length;n++){i=(i+step+sections.length)%sections.length;const s=sections[i];if(accepted&&s.dataset.selected!=='true')continue;for(const k of ['source','aspect','mode'])document.getElementById(k).value=s.dataset[k];render();return;}}
for(const [id,step,accepted] of [['previous',-1,false],['next',1,false],['accepted',1,true]])document.getElementById(id).addEventListener('click',()=>move(step,accepted));
render();</script>"""
    (a.pilot/a.output).write_text(page)


if __name__=='__main__':main()
