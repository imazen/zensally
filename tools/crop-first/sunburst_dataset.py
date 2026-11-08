#!/usr/bin/env python3
"""Prepare verified lossless image blobs and metadata; never perform Git/network actions."""
import argparse,csv,hashlib,io,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write_once(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('existing dataset artifact differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('comparison',type=Path)
    p.add_argument('dataset',type=Path);p.add_argument('collection')
    p.add_argument('--native-archive',required=True)
    a=p.parse_args()
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,90}',a.collection):raise ValueError('invalid collection')
    jobs=json.loads((a.source/'plan.json').read_text())
    review=json.loads((a.source/'review.json').read_text())
    summary=json.loads((a.comparison/'summary.json').read_text())
    compressed={r['id']:r for r in summary['records']}
    if len(jobs)!=len(compressed) or {j['id'] for j in jobs}!=set(compressed):raise ValueError('compression must cover every planned source')
    prepared=[]
    for job in jobs:
        identity=job['id']
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,90}',identity):raise ValueError('invalid identity')
        native=a.source/(identity+'.png');record=json.loads((a.source/(identity+'.json')).read_text())
        row=compressed[identity];v=row['variants']['lossless'];image=a.comparison/identity/'lossless.jxl'
        if record['png_sha256']!=sha(native) or row['png_sha256']!=record['png_sha256']:raise ValueError('native source differs')
        if sha(image)!=v['sha256'] or not v['lossless_exact'] or not v['independent_exact'] or v['decoded_rgb_sha256']!=row['rgb_sha256']:raise ValueError('lossless evidence differs')
        if record['id']!=identity or record['prompt']!=job['prompt']:raise ValueError('native generation metadata differs')
        if identity not in review['notes']:raise ValueError('missing source inspection')
        record.update(storage={'format':'jxl','mode':'lossless','effort':7,'bytes':image.stat().st_size,'sha256':v['sha256'],'rgb_sha256':row['rgb_sha256'],'owned_and_independent_pixel_exact':True,'original_png_archive':a.native_archive.rstrip('/')+'/'+identity+'.png','embedded_png_provenance_retained_in_original_archive':True},study=a.collection,source_id='ai-'+a.collection+'-'+identity,coordinator_inspection=review['notes'][identity],human_rated=False,training_eligible=False,face_safety_certified=False)
        if identity in review.get('observed_primary_people',{}):record['coordinator_observed_primary_people']=review['observed_primary_people'][identity]
        if identity in review.get('brand_rendering_issues',{}):record['brand_rendering_issue']=review['brand_rendering_issues'][identity]
        prepared.append((identity,image,record,row))
    output=a.dataset/'ai-originals'/a.collection;output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for identity,image,record,r in prepared:
        target=output/(identity+'.jxl');write_once(target,image.read_bytes())
        write_once(output/(identity+'.json'),(json.dumps(record,indent=2)+'\n').encode())
        rows.append({'id':identity,'file':str(target.relative_to(a.dataset)),'width':r['width'],'height':r['height'],'bytes':image.stat().st_size,'sha256':record['storage']['sha256'],'native_png_sha256':r['png_sha256'],'rgb_sha256':r['rgb_sha256'],'scenario_family':record['scenario_family'],'training_eligible':'false'})
    text=io.StringIO();w=csv.DictWriter(text,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    write_once(output/'manifest.tsv',text.getvalue().encode())
    print(json.dumps({'images':len(rows),'bytes':sum(r['bytes'] for r in rows),'directory':str(output)}))


if __name__=='__main__':main()
