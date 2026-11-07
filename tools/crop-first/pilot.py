#!/usr/bin/env python3
"""Render a bounded own-source annotation pilot; no labels are training-admitted.

Uses owned Rust rendering. Pillow is only an independent readback oracle.
The intentionally unfiltered proposals may be bad crops, never recommendations.
"""
import argparse,csv,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifests',type=Path);p.add_argument('executable',type=Path)
    p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('quarantine',type=Path)
    a=p.parse_args()
    from PIL import Image # independent decoder only; never writes candidate pixels
    inventory=a.manifests/'train.tsv';raw=inventory.read_bytes()
    quarantine=json.loads(a.quarantine.read_text())['training_quarantine_ids']
    rows={r['id']:r for r in csv.DictReader(raw.decode().splitlines(),delimiter='\t')}
    ids=['1400','1600']
    assert not set(ids)&set(quarantine)
    a.inputs.mkdir(exist_ok=False);a.output.mkdir(parents=True,exist_ok=False)
    protocol={'frozen_at':datetime.now(timezone.utc).isoformat(),'source_ids':ids,
        'source_role':'train','source_manifest_sha256':sha(raw),'tasks_per_source':[
            {'mode':'minimal','ratio':[9,16]},{'mode':'maximal','ratio':[1,1]}],
        'supervision':'vlm_preference','admission':'unreviewed','training_eligible':False,
        'face_safety_certified':False,'proposal_status':'unfiltered annotation examples, never crop recommendations',
        'renderer_sha256':sha(a.executable.read_bytes()),'human_calibration':False}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sources=[];tasks=[];oracle=[];links=[]
    for sid in ids:
        r=rows[sid];assert r['split']=='train'
        url=r['png_v3_sdr_url'];links.append({'text':f'Canonical PNG source {sid}','href':url})
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 zensally-crop-pilot'}),timeout=120) as response:data=response.read()
        source=a.inputs/f'{sid}.png';source.write_bytes(data)
        record={'id':sid,'role':'train','png_sha256':sha(data),'canonical_url':url,
                'parent_original_sha256':r['sha256'],'width':int(r['width']),'height':int(r['height'])}
        sources.append(record)
        out=a.output/sid
        subprocess.run([str(a.executable),str(source),str(out),sid],check=True)
        with Image.open(source) as im:
            assert im.size==(record['width'],record['height'])
            if 'A' in im.getbands():assert im.getchannel('A').getextrema()==(255,255)
        for path in sorted(out.glob('*.png')):
            with Image.open(path) as im:
                im.load();rgb=im.convert('RGB').tobytes()
                assert sha(rgb)==path.with_suffix('.rgb-sha256').read_text().strip(),path
                oracle.append({'file':str(path.relative_to(a.output)),'dimensions':list(im.size),'rgb_sha256':sha(rgb)})
        by_mode={}
        for c in csv.DictReader((out/'candidates.tsv').open(),delimiter='\t'):
            by_mode.setdefault(c['mode'],[]).append(c)
        def image_record(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
        for mode,candidates in by_mode.items():
            task={'task_id':f'{sid}-{mode}','source_id':sid,'source_role':'train','mode':mode,
                'target_ratio':[int(candidates[0]['ratio_w']),int(candidates[0]['ratio_h'])],
                'source':image_record(out/'source.png'),'candidates':[
                    {**image_record(out/c['file']),'id':f"{sid}-{mode}-{c['candidate']}",
                     'source_box_xywh':[int(c[k]) for k in ['x','y','w','h']]} for c in candidates]}
            (out/f'{mode}-task.json').write_text(json.dumps(task,indent=2)+'\n');tasks.append(task)
    (a.inputs/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    (a.inputs/'links.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in links))
    (a.output/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (a.output/'oracle.json').write_text(json.dumps(oracle,indent=2)+'\n')
    print(json.dumps({'sources':len(sources),'tasks':len(tasks),'independently_decoded_previews':len(oracle),'admission':'unreviewed','training_eligible':False}))


if __name__=='__main__':main()
