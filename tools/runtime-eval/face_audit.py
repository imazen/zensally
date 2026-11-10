#!/usr/bin/env python3
"""Match fixed visual face-center probes to local detections, without fitting."""
import argparse,json
from pathlib import Path

def matched(points,boxes):
 edges=[[i for i,(x0,y0,x1,y1) in enumerate(boxes) if x0<=x<=x1 and y0<=y<=y1] for x,y in points];owners={}
 def visit(point,seen):
  for box in edges[point]:
   if box in seen:continue
   seen.add(box)
   if box not in owners or visit(owners[box],seen):owners[box]=point;return True
  return False
 for i in range(len(points)):visit(i,set())
 return sorted(owners.values())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();doc=json.loads((a.root/'manifest.json').read_text());sources={r['id']:r for r in doc['images']};annotation=json.loads(Path(__file__).with_name('face_points.json').read_text());report={'annotation':annotation,'metric':'one-to-one center inside predicted rectangle; not IoU recall, human truth or head containment','models':{}}
 for group in ['reference-eval','rust-eval','rust-front-fixed','tiled-sparse-eval']:
  for model in ([a.root/group] if group=='tiled-sparse-eval' else (a.root/group).iterdir()):
   path=model/'results.json'
   if not path.exists():continue
   raw=json.loads(path.read_text());rows=raw if isinstance(raw,list) else raw['rows'];rows={r['id']:r['result'] for r in rows};out=[]
   if not any('faces' in r or 'detections' in r for r in rows.values()):continue
   if group=='reference-eval' and 'blaze_face' not in model.name:continue
   for id,entry in annotation['sources'].items():
    v=rows[id];s=sources[id];boxes=[]
    for f in v.get('faces',[]):boxes.append([x/100 for x in f['box_percent']])
    for d in v.get('detections',[]):
     x,y,w,h=d['xywh'];boxes.append([x/s['width'],y/s['height'],(x+w)/s['width'],(y+h)/s['height']])
    points=[[x/entry['size'][0],y/entry['size'][1]] for x,y in entry['points']];hits=matched(points,boxes)
    out.append({'id':id,'marked':len(points),'detections':len(boxes),'matched_centers':len(hits),'missed_point_indices':[i for i in range(len(points)) if i not in hits]})
   name='front-fixed' if group=='rust-front-fixed' else 'sparse-tiles5' if group=='tiled-sparse-eval' else model.name
   report['models'][name]={'scenes':out,'marked':sum(r['marked'] for r in out),'matched_centers':sum(r['matched_centers'] for r in out),'detections':sum(r['detections'] for r in out)}
 with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:{key:v[key] for key in ['marked','matched_centers','detections']} for k,v in report['models'].items()}))
if __name__=='__main__':main()
