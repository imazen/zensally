#!/usr/bin/env python3
"""Compare independent Terra/Sol decisions and supplied-head geometry, not recall."""
import argparse
import json
from pathlib import Path
from subject_geometry import inspect

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','output']:p.add_argument(key,type=Path)
    a=p.parse_args();rows=[]
    for path in sorted((a.run/'real-sol').glob('openimages-*.json')):
        sol=json.loads(path.read_text());terra=json.loads((a.run/'real-terra'/path.name).read_text())
        task=json.loads((a.run/'real-tasks'/(path.stem.split('-')[-1]+'.json')).read_text())
        row=dict(id=task['id'],terra_valid=terra['annotation_valid'],sol_valid=sol['annotation_valid'])
        if sol['annotation_valid'] and terra['annotation_valid']:
            left=json.loads((Path(terra['entry'])/'zones.json').read_text());right=json.loads((Path(sol['entry'])/'zones.json').read_text())
            row.update(decisions=len(left['crops']),action_agreement=sum(x['action']==y['action'] for x,y in zip(left['crops'],right['crops'])))
            for name,zones in [('terra',left),('sol',right)]:
                d=inspect(zones,task['width'],task['height'])
                row[name]=dict(options=d['options'],raw_geometry_failures=d['raw_geometry_failures'],projected_geometry_failures=d['projected_geometry_failures'],face_rois=sum(v['kind']=='face_head' for v in zones['rois']))
        rows.append(row)
    result=dict(sources=len(rows),structurally_comparable=sum('decisions' in v for v in rows),action_agreement=sum(v.get('action_agreement',0) for v in rows),
        decisions=sum(v.get('decisions',0) for v in rows),rows=rows,complete_face_coverage_certified=False)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

