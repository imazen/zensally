#!/usr/bin/env python3
"""Render percentage proposals with owned codecs; retain raw/projected geometry."""
import argparse,csv,hashlib,html,json,math,subprocess
from pathlib import Path
from expanded import HEADER

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('packets',type=Path);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('renderer',type=Path);a=p.parse_args()
    from PIL import Image # independent readback only
    a.output.mkdir(parents=True,exist_ok=False);packets=json.loads(a.packets.read_text());records=[];readback=[];parts=[]
    for packet in packets:
        source=packet['source'];sid=source['id'];sw,sh=source['width'],source['height'];src=a.inputs/(sid+'.png')
        if hashlib.sha256(src.read_bytes()).hexdigest()!=source['png_sha256']:raise ValueError('original source hash changed')
        rows=[];items=[]
        for crop in packet['zones']['crops']:
            rw,rh=map(int,crop['ratio'].split(':'));k0=min(sw//rw,sh//rh)
            for i,option in enumerate(crop['options'] if crop['action']=='crop' else []):
                b=option['box'];cx=(b['left']+b['right'])/200*sw;cy=(b['top']+b['bottom'])/200*sh
                k=k0 if crop['mode']=='minimal' else max(math.ceil(k0*.5),min(k0,math.floor(min((b['right']-b['left'])/100*sw/rw,(b['bottom']-b['top'])/100*sh/rh))))
                w,h=k*rw,k*rh;x=max(0,min(sw-w,round(cx-w/2)));y=max(0,min(sh-h,round(cy-h/2)))
                name=f'{rw}x{rh}-{crop["mode"]}-D{i+1:02}.png';tid=f'{sid}-{rw}x{rh}-{crop["mode"]}'
                row=[tid,sid,crop['mode'],rw,rh,sw,sh,f'D{i+1:02}',x,y,w,h,name,'direct-percentage-proposal'];rows.append(row)
                missed=[]
                for roi in packet['zones']['rois']:
                    if roi['kind']=='face_head':
                        f=roi['box'];bounds=[math.floor(f['left']/100*sw),math.floor(f['top']/100*sh),math.ceil(f['right']/100*sw),math.ceil(f['bottom']/100*sh)]
                        if not(x<=bounds[0] and y<=bounds[1] and x+w>=bounds[2] and y+h>=bounds[3]):missed.append(roi['id'])
                item={'task_id':tid,'mode':crop['mode'],'ratio':crop['ratio'],'raw_percent_box':b,'projected_xywh':[x,y,w,h],'file':name,'known_proposed_face_rois_not_contained':missed,'teacher_reason':option['reason'],'admission':'unreviewed','training_eligible':False,'face_safety_certified':False};items.append(item)
        table=a.output/(sid+'-render.tsv')
        with table.open('x') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(HEADER);writer.writerows(rows)
        out=a.output/sid
        subprocess.run([str(a.renderer),str(src),str(out),sid,str(table),'768'],check=True)
        for path in sorted(out.glob('*.png')):
            with Image.open(path) as image:
                image.load();readback.append({'file':str(path.relative_to(a.output)),'width':image.width,'height':image.height,'rgb_sha256':hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),'png_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        roi_svg=f'<svg viewBox="0 0 {sw} {sh}"><image href="{sid}/source.png" width="{sw}" height="{sh}"/>'
        for roi in packet['zones']['rois']:
            b=roi['box'];color='#f66' if roi['kind']=='face_head' else '#ffdb4d' if roi['priority']=='primary' else '#7df'
            roi_svg+=f'<rect x="{b["left"]*sw/100}" y="{b["top"]*sh/100}" width="{(b["right"]-b["left"])*sw/100}" height="{(b["bottom"]-b["top"])*sh/100}" fill="none" stroke="{color}" stroke-width="{sw/250}"><title>{html.escape(roi["id"]+": "+roi["description"])}</title></rect>'
        roi_svg+='</svg>'
        parts.append(f'<section><h2>Source {sid} · {source["role"]}</h2><div class="hero">{roi_svg}<div><h3>Regions of interest</h3>'+''.join(f'<p>{html.escape(r["id"])} ({r["kind"]}, {r["priority"]}): {html.escape(json.dumps(r["box"]))}%</p>' for r in packet['zones']['rois'])+'</div></div>')
        for crop in packet['zones']['crops']:
            matching=[r for r in items if r['ratio']==crop['ratio'] and r['mode']==crop['mode']]
            parts.append(f'<details><summary>{crop["ratio"]} {crop["mode"]} · {crop["action"]}</summary><p>{html.escape(crop["reason"])}</p><div class="images">')
            for item in matching:
                x,y,w,h=item['projected_xywh'];overlay=f'<svg viewBox="0 0 {sw} {sh}"><image href="{sid}/source.png" width="{sw}" height="{sh}"/><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#3f8" stroke-width="{sw/180}"/></svg>'
                parts.append(f'<figure>{overlay}<figcaption>Projected source rectangle</figcaption></figure><figure><img loading="lazy" src="{sid}/{item["file"]}"><figcaption>{html.escape(item["teacher_reason"])}<br>Raw %: {html.escape(json.dumps(item["raw_percent_box"]))}<br>Integer xywh: {item["projected_xywh"]}<br>Proposed face ROIs outside: {html.escape(str(item["known_proposed_face_rois_not_contained"]))}</figcaption></figure>')
            if not matching:parts.append(f'<p>Keep the full source and pad. {html.escape(crop["reason"])}</p>')
            parts.append('</div></details>')
        parts.append('</section>');records.append({**packet,'rendered':items})
    (a.output/'annotations.json').write_text(json.dumps(records,indent=2)+'\n');(a.output/'readback.json').write_text(json.dumps(readback,indent=2)+'\n')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Direct ROI and crop-zone proposals</title><style>body{font:16px system-ui;background:#15171b;color:#eee;margin:24px}section{border-top:1px solid #777;margin-top:32px}.hero,.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}img,svg{width:100%;height:460px;object-fit:contain}figure{margin:0}summary{padding:14px;cursor:pointer}figcaption{font-size:14px}@media(max-width:700px){.hero,.images{grid-template-columns:1fr}}</style><h1>Direct ROI and crop-zone proposals</h1><p>One image-attached API request per source returns ROI boxes and all 14 crop decisions in original-image percentages. Yellow: primary ROI; cyan: other ROIs; red: proposed face/head. These are unreviewed proposals, not human quality or face-safety certification. Crops below use owned pixels and exact integer target aspect. Raw and projected rectangles remain separate; Terra has not judged these rendered pixels. Minimal keeps maximum area; Maximal projection retains the existing 0.5–1 scale range. Hover ROI rectangles for descriptions.</p>'''+''.join(parts)
    (a.output/'index.html').write_text(page)
    summary={'sources':len(records),'requests':sum(len(r['zones']['crops']) for r in records),'rendered_crops':sum(len(r['rendered']) for r in records),'readback_pngs':len(readback),'known_proposed_face_roi_violations':sum(bool(i['known_proposed_face_rois_not_contained']) for r in records for i in r['rendered']),'training_eligible':False,'face_safety_certified':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
