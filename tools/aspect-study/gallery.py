#!/usr/bin/env python3
"""Build an interactive review of real extracted crops and frozen metrics."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('analysis',type=Path);p.add_argument('visuals',type=Path);a=p.parse_args()
selection=json.loads((a.analysis/'gallery-selection.json').read_text())
methods=json.loads((a.analysis/'summary.json').read_text())['methods']
extra=a.visuals/'real-selection.json'
if extra.exists():selection+=json.loads(extra.read_text())
# Re-rendering must reproduce all frozen per-case numbers, even when image
# order and workspace history differ. Never substitute these for full results.
original={}
for kind in ['natural','stress']:
 wanted={(r['id'],r['case']) for r in selection if r['kind']==kind}
 for shard in range(4):
  for r in csv.DictReader((a.run/f'{kind}-{shard}'/'metrics.tsv').open(),delimiter='\t'):
   if (r['id'],r['case']) in wanted:original[(kind,r['id'],r['case'],r['method'])]=r
cases=[]
for kind in ['natural','stress','real']:
 metrics={}
 for r in csv.DictReader((a.visuals/kind/'metrics.tsv').open(),delimiter='\t'):
  key=(kind,r['id'],r['case'],r['method'])
  if kind=='real' or key in original:
   if kind!='real':assert r==original[key],f'Render order changed frozen output: {key}'
   if kind=='real':r['mae']=None
   metrics[key]=r
 crops={}
 for r in csv.DictReader((a.visuals/kind/'crops.tsv').open(),delimiter='\t'):
  if kind=='real':
   for f in ['retention','oracle_retention','regret']:r[f]=None
  crops[(r['id'],r['case'],r['method'],r['aspect'],r['mode'])]=r
 for item in selection:
  if item['kind']!=kind:continue
  ident=item['id'];case=item['case'];directory=a.visuals/kind/'gallery'/f'{ident}-{case}'
  assert (directory/'source.png').exists(),directory
  method_rows={m:metrics[(kind,ident,case,m)] for m in methods}
  first=next(iter(method_rows.values()));methods={}
  for method,row in method_rows.items():
   choices={}
   for aspect in ['1:1','16:9','9:16']:
    for mode in ['minimal','maximal']:
     choices[f'{aspect}/{mode}']=crops[(ident,case,method,aspect,mode)]
   methods[method]={'metrics':row,'crops':choices}
  cases.append({**item,'width':int(first['width']),'height':int(first['height']),'directory':str(directory.relative_to(a.visuals)),'methods':methods})
(a.visuals/'review.json').write_text(json.dumps(cases,indent=2)+'\n')
html='''<!doctype html><meta charset="utf-8"><title>Saliency geometry and crop review</title>
<style>
body{font:16px system-ui;margin:24px;background:#17191d;color:#eee}a{color:#8dc9ff}button,select,input{font:inherit;margin:6px;padding:5px}header{position:sticky;top:0;background:#17191def;z-index:2;padding:8px;border-bottom:1px solid #555}.context{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1200px}.context img{width:100%;height:220px;object-fit:contain;background:#292d33}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:18px}.card{background:#252a32;padding:14px;border-radius:8px}.scene-scroll{max-height:380px;overflow:auto}.frame{position:relative;width:100%;background:#333}.frame img,.frame svg{position:absolute;inset:0;width:100%;height:100%}.heat{opacity:.5}.crop{display:block;max-width:100%;height:220px;object-fit:contain;margin:auto}small{color:#bcc6d5}.numbers{font:13px ui-monospace;line-height:1.7}h2{margin:5px 0}footer{margin:28px 0;max-width:1100px;line-height:1.5}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(240px,1fr))}}@media(max-width:650px){.grid{grid-template-columns:1fr}}
</style>
<header><h2>Saliency geometry and overlapping crop refinement</h2><button id="previous">Previous</button><select id="case"></select><button id="next">Next</button><br>
<label>Crop <select id="aspect"><option>1:1</option><option>16:9</option><option>9:16</option></select></label>
<label>Mode <select id="mode"><option value="minimal">Minimal: largest crop</option><option value="maximal">Maximal: tight crop</option></select></label>
<label>Heatmap opacity <input id="opacity" type="range" min="0" max="1" step=".05" value=".5"></label><p id="details"></p></header>
<p>Corrected crop placement maximizes retained heatmap mass; the original study remains preserved. Red: predicted foreground; blue: background. Yellow outlines show the exact crop. These are actual extracted crops, not CSS crop approximations. Examples include deliberately selected improvements and regressions; use the full-study tables for aggregate quality.</p>
<div class="context"><figure><img id="source"><figcaption>Source <a id="original" target="_blank" rel="noopener">Open full source</a></figcaption></figure><figure><img id="mask"><figcaption>Human foreground mask (projected scoring grid)</figcaption></figure></div><div class="grid" id="cards"></div>
<footer>Human masks are a subject-retention proxy, not an aesthetic crop rating. Synthetic canvases preserve the source photo's proportions on a constant background; they are labeled separately from natural images. Model input remains 256×256. No weights were fitted. Source-level calibration selected policies separately from evaluation. <a href="https://saliencydetection.net/duts/">DUTS dataset</a> · <a href="https://arxiv.org/abs/1908.07274">Global/local saliency research</a></footer>
<script>
const cases=DATA;
const select=document.getElementById('case');cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.kind}: ${c.id} / ${c.case} (${c.selection})`;select.append(o)});
function render(){const c=cases[+select.value],aspect=document.getElementById('aspect').value,mode=document.getElementById('mode').value,key=`${aspect}/${mode}`;
document.getElementById('details').textContent=`${c.kind} · ${c.width}×${c.height} · selected as ${c.selection}${c.method?' for '+c.method:''}`;
document.getElementById('source').src=`${c.directory}/source.png`;document.getElementById('original').href=c.source_url||`${c.directory}/source.png`;document.getElementById('mask').parentElement.style.display=c.kind==='real'?'none':'';if(c.kind!=='real')document.getElementById('mask').src=`${c.directory}/mask.png`;
const cards=document.getElementById('cards');cards.replaceChildren();
for(const [method,value] of Object.entries(c.methods)){const r=value.crops[key],m=value.metrics,card=document.createElement('section');card.className='card';const ratio=aspect.replace(':','x');
card.innerHTML=`<h3>${method}</h3><small>${m.passes} model passes · MAE ${m.mae===null?'unlabelled':(+m.mae).toFixed(4)}</small><button class="locate">Locate crop in heatmap</button><div class="scene-scroll"><div class="frame" style="aspect-ratio:${c.width}/${c.height}"><img src="${c.directory}/source.png"><img class="heat" src="${c.directory}/${method}-map.png"><svg viewBox="0 0 ${c.width} ${c.height}"><rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="none" stroke="yellow" stroke-width="${Math.max(c.width,c.height)/200}"/></svg></div></div><img class="crop" src="${c.directory}/${method}-${ratio}-${mode}.png"><p class="numbers">Retained foreground: ${r.retention===null?'unlabelled':(100*+r.retention).toFixed(1)+'%'}<br>Best at same size: ${r.oracle_retention===null?'unlabelled':(100*+r.oracle_retention).toFixed(1)+'%'}<br>Crop: x=${r.x}, y=${r.y}, ${r.w}×${r.h}<br>Source area retained: ${(100*r.w*r.h/(c.width*c.height)).toFixed(1)}%</p>`;cards.append(card);if(['stretch','letterbox','letterbox-mean','letterbox-blur','letterbox-edge-blur'].includes(method)){const d=document.createElement('details');d.innerHTML=`<summary>Show exact model input (256×256)</summary><img class="crop" src="${c.directory}/${method}-input.png">`;card.append(d)}card.querySelector('.locate').onclick=()=>{const sc=card.querySelector('.scene-scroll');sc.scrollTop=Math.max(0,((+r.y)+(+r.h)/2)/c.height*sc.scrollHeight-sc.clientHeight/2)}}
opacity();}
function opacity(){document.querySelectorAll('.heat').forEach(e=>e.style.opacity=document.getElementById('opacity').value)}
for(const id of ['case','aspect','mode'])document.getElementById(id).onchange=render;document.getElementById('opacity').oninput=opacity;
document.getElementById('previous').onclick=()=>{select.value=(+select.value+cases.length-1)%cases.length;render()};document.getElementById('next').onclick=()=>{select.value=(+select.value+1)%cases.length;render()};render();
</script>'''
(a.visuals/'index.html').write_text(html.replace('DATA',json.dumps(cases).replace('</','<\\/')))
print(json.dumps({'cases':len(cases),'frozen_method_rows_verified':len(original),'url_file':str(a.visuals/'index.html')}))
