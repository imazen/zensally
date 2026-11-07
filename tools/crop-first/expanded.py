#!/usr/bin/env python3
"""Frozen corpus expansion and rendered direct-proposal study. No training admission."""
import argparse,csv,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from proposals import candidate_pool

RATIOS=[(1,1),(9,16),(16,9),(1,4),(4,1),(1,8),(8,1)]
HEADER=['task_id','source_id','mode','ratio_w','ratio_h','source_w','source_h','candidate','x','y','w','h','file','generators']
TEXT={'5000','5200','5300','6000','6800','7000','8000','8100'}

def sha(data):return hashlib.sha256(data).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,data):Path(p).write_text(json.dumps(data,indent=2)+'\n')
def kind(group):
    if group in TEXT:return 'text_or_screenshot'
    if group in {'2200','2400','6600','9000','9094'}:return 'art_or_illustration'
    if group=='9226':return 'synthetic_product'
    return 'photograph_or_art_photo'

def task_id(sid,ratio,mode):return f'{sid}-{ratio[0]}x{ratio[1]}-{mode}'
def progress(root,stage,**fields):
    previous=load(root/'progress.json') if (root/'progress.json').exists() else {}
    if stage!='failed':fields.setdefault('error',None)
    save(root/'progress.json',{**previous,'checked_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'stage':stage,**fields})

def freeze(a):
    root=a.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():raise ValueError('selection already frozen')
    quarantine=set(load(a.quarantine)['training_quarantine_ids']);by_role={};hashes={}
    for role in ['train','validate']:
        p=a.manifests/f'{role}.tsv';raw=p.read_bytes();hashes[role]=sha(raw)
        rows=list(csv.DictReader(raw.decode().splitlines(),delimiter='\t'))
        by_role[role]=[r for r in rows if int(r['width'])*int(r['height'])<=32_000_000 and min(int(r['width']),int(r['height']))>=128 and (role!='train' or r['id'] not in quarantine)]
    def rank(r):return sha(('zensally-expanded-v1|'+r['id']).encode())
    train=by_role['train'];groups=sorted({r['content_class'].split('-')[0] for r in train})
    chosen=[min([r for r in train if r['content_class'].split('-')[0]==g],key=rank) for g in groups]
    for group,metric in [('2000',lambda r:int(r['height'])/int(r['width'])),('1400',lambda r:int(r['width'])/int(r['height'])),('9226',lambda r:int(r['height'])/int(r['width']))]:
        candidates=[r for r in train if r['content_class'].split('-')[0]==group and r not in chosen]
        chosen.append(max(candidates,key=lambda r:(metric(r),rank(r))))
    validate=by_role['validate'];val=[]
    for sid in ['8023','5015','3313']:val.append(next(r for r in validate if r['id']==sid))
    for group in ['2000','1400','1200','9226','8100']:val.append(min([r for r in validate if r['content_class'].split('-')[0]==group and r not in val],key=rank))
    chosen+=val
    assert len(chosen)==32 and len({r['sha256'] for r in chosen})==32
    records=[]
    for r in chosen:
        group=r['content_class'].split('-')[0]
        records.append({'id':r['id'],'role':r['split'],'content_group':group,'content_kind':kind(group),'width':int(r['width']),'height':int(r['height']),'canonical_url':r['png_v3_sdr_url'],'parent_original_sha256':r['sha256']})
    protocol={'frozen_at':datetime.now(timezone.utc).isoformat(),'sources':32,'roles':{'train':24,'validate':8},'source_manifest_sha256':hashes,'quarantine_sha256':sha(a.quarantine.read_bytes()),'ratios':RATIOS,'modes':['minimal','maximal'],'required_tasks':448,'sources_per_batch':2,'human_calibration':False,'admission':'unreviewed','training_eligible':False,'face_safety_certified':False,'source_preview_long_edge':768,'candidate_preview_pixel_cap':262144,'candidate_preview_long_edge_cap':1536,'note':'No TEST sources. Known cross-role train relatives excluded; no complete duplicate/face audit. Real canonical originals; source-aspect and target-aspect strata are distinct.'}
    save(root/'protocol.json',protocol);save(root/'selection.json',records)
    a.inputs.mkdir(parents=True,exist_ok=False)
    save(a.inputs/'source-manifests.json',hashes)
    (a.inputs/'links.jsonl').write_text(''.join(json.dumps({'text':f'Canonical PNG source {r["id"]}','href':r['canonical_url']})+'\n' for r in records))
    progress(root,'frozen',sources_complete=0,sources_required=32,tasks_complete=0,tasks_required=448)
    print(json.dumps({'frozen_sources':len(records),'tasks':448,'groups':len(groups)}))

def write_table(path,source,candidates):
    with path.open('x') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(HEADER)
        for t,c in candidates:
            rw,rh=t['target_ratio'];mode=t['mode'];tid=t['task_id'];file=f'{rw}x{rh}-{mode}-{c["id"]}.png'
            w.writerow([tid,source['id'],mode,rw,rh,source['width'],source['height'],c['id'],c['x'],c['y'],c['w'],c['h'],file,','.join(c['generators'])])

def planned(source):
    return [{'task_id':task_id(source['id'],r,m),'target_ratio':list(r),'mode':m} for r in RATIOS for m in ['minimal','maximal']]

def resolve(a):
    from urllib.parse import quote
    inventory=subprocess.run(['git','-C',str(a.variant_root),'ls-files','png-v3/'],check=True,capture_output=True,text=True).stdout
    commit=subprocess.run(['git','-C',str(a.variant_root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    index={}
    for name in inventory.splitlines():
        sid=Path(name).name.split('_',1)[0]
        if name.endswith('.sdr.png'):index.setdefault(sid,[]).append(name)
    records=[]
    for r in load(a.output/'selection.json'):
        paths=index.get(r['id'],[])
        if len(paths)!=1:raise ValueError(f"expected one tracked SDR PNG for {r['id']}, got {len(paths)}")
        path=paths[0]
        url='https://codec-corpus.r2.imazen.org/imazen-26-png-v3/'+quote(path.removeprefix('png-v3/'),safe='/')
        records.append({**r,'manifest_png_url':r['canonical_url'],'canonical_url':url,'variant_path':path})
    save(a.output/'variant-resolution.json',{'variant_commit':commit,'tracked_inventory_sha256':sha(inventory.encode()),'sources':records})
    print(json.dumps({'resolved_sources':len(records),'urls_differ':sum(r['manifest_png_url']!=r['canonical_url'] for r in records)}))


def acquire(a):
    from PIL import Image # independent readback only
    records=load(a.output/'variant-resolution.json')['sources'];done=[];oracle=[]
    progress(a.output,'acquiring',sources_complete=0,sources_required=len(records))
    for r in records:
        path=a.inputs/f'{r["id"]}.png'
        if path.exists():raise ValueError('fresh acquisition refuses to replace source bytes')
        with urlopen(Request(r['canonical_url'],headers={'User-Agent':'Mozilla/5.0 zensally-crop-study'}),timeout=120) as response:data=response.read()
        path.write_bytes(data)
        dimensions=json.loads(subprocess.run([str(a.executable),'--inspect',str(path)],check=True,capture_output=True,text=True).stdout)
        r={**r,'manifest_width':r['width'],'manifest_height':r['height'],**dimensions,'png_sha256':sha(data)}
        t=planned(r)[0];c=candidate_pool(r['width'],r['height'],'minimal',target_ratio=(1,1))[0]
        table=a.output/f'{r["id"]}-preview.tsv';write_table(table,r,[(t,c)])
        out=a.output/'sources'/r['id'];out.parent.mkdir(exist_ok=True)
        subprocess.run([str(a.executable),str(path),str(out),r['id'],str(table),'768'],check=True)
        with Image.open(path) as im:
            assert im.size==(r['width'],r['height'])
            assert sha(im.convert('RGB').tobytes())==r['rgb_sha256']
            if 'A' in im.getbands():assert im.getchannel('A').getextrema()==(255,255)
        for p in out.glob('*.png'):
            with Image.open(p) as im:
                im.load();assert sha(im.convert('RGB').tobytes())==p.with_suffix('.rgb-sha256').read_text().strip()
                oracle.append({'file':str(p.relative_to(a.output)),'dimensions':list(im.size)})
        done.append(r);save(a.inputs/'sources.json',done);progress(a.output,'acquiring',sources_complete=len(done),sources_required=len(records),last_source=r['id'])
    save(a.output/'source-oracle.json',oracle)
    batches=a.output/'batches';batches.mkdir()
    for i in range(0,len(done),2):
        subset=[]
        for r in done[i:i+2]:
            p=a.output/'sources'/r['id']/'source.png'
            subset.append({**r,'source':{'path':str(p),'sha256':sha(p.read_bytes())},'tasks':planned(r)})
        save(batches/f'{i//2:02}-sources.json',subset)
    progress(a.output,'sources-ready',sources_complete=len(done),sources_required=len(records),proposal_batches_complete=0,proposal_batches_required=16,tasks_required=448)
    print(json.dumps({'sources':len(done),'readback_previews':len(oracle),'batches':16,'tasks':448}))

def render(a):
    from PIL import Image
    sources={r['id']:r for r in load(a.inputs/'sources.json')};batch=load(a.batch);raw=load(a.proposals)
    required=[t for r in batch for t in r['tasks']]
    if [r['task_id'] for r in raw]!=[t['task_id'] for t in required]:raise ValueError('missing/extra/reordered proposal tasks')
    hints=dict(zip([r['task_id'] for r in raw],raw));tasks=[];oracle=[]
    for source in batch:
        sid=source['id'];r=sources[sid];rows=[];metadata=[]
        for t in source['tasks']:
            v=hints[t['task_id']]
            if set(v)!={'task_id','action','proposals','reason'} or v['action'] not in ['crop','contain','unchanged'] or not isinstance(v['reason'],str):raise ValueError('invalid direct proposal record')
            if len(v['proposals'])>2:raise ValueError('at most two direct proposals')
            if (v['action']=='contain') != (len(v['proposals'])==0):raise ValueError('contain must have no crop proposals')
            pool=candidate_pool(r['width'],r['height'],t['mode'],v['proposals'],target_ratio=t['target_ratio'])
            # Render the center baseline and every direct proposal. The first,
            # frozen baseline is an explicit control, not an endpoint pair.
            selected=[c for c in pool if c['id']=='C01' or 'vision-proposal' in c['generators']]
            rows.extend((t,c) for c in selected);metadata.append((t,v,selected))
        folder=a.output/'crops'/sid;folder.parent.mkdir(exist_ok=True)
        table=a.output/f'{sid}-crops.tsv';write_table(table,r,rows)
        subprocess.run([str(a.executable),str(a.inputs/f'{sid}.png'),str(folder),sid,str(table),'768'],check=True)
        for path in folder.glob('*.png'):
            with Image.open(path) as im:
                im.load();assert sha(im.convert('RGB').tobytes())==path.with_suffix('.rgb-sha256').read_text().strip()
                oracle.append({'file':str(path.relative_to(a.output)),'dimensions':list(im.size)})
        for t,v,cs in metadata:
            def image(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
            candidates=[{**image(folder/f'{t["target_ratio"][0]}x{t["target_ratio"][1]}-{t["mode"]}-{c["id"]}.png'),'id':t['task_id']+'-'+c['id'],'display_id':c['id'],'source_box_xywh':[c[k] for k in ['x','y','w','h']],'generators':c['generators']} for c in cs]
            candidates.sort(key=lambda c:sha((t['task_id']+'|'+c['id']).encode()))
            tasks.append({**t,'source_id':sid,'source_role':r['role'],'content_kind':r['content_kind'],'source':image(folder/'source.png'),'candidates':candidates,'direct_action':v['action'],'direct_reason':v['reason'],'admission':'unreviewed','training_eligible':False,'face_safety_certified':False})
    name=a.batch.name.replace('-sources.json','');save(a.output/'batches'/f'{name}-tasks.json',tasks);save(a.output/'batches'/f'{name}-oracle.json',oracle)
    print(json.dumps({'rendered_tasks':len(tasks),'candidate_previews':sum(len(t['candidates']) for t in tasks),'readback_images':len(oracle)}))

def status(a):
    batches=a.output/'batches'
    source_count=len(load(a.inputs/'sources.json'))
    proposals=[load(p) for p in batches.glob('??-proposals.json')]
    rendered=[load(p) for p in batches.glob('??-tasks.json')]
    labels=[load(p)['records'] for p in batches.glob('??-labels.json')]
    progress(a.output,'annotating' if sum(map(len,labels))<448 else 'annotations-complete',sources_complete=source_count,
        proposal_batches_complete=len(proposals),proposal_batches_required=16,proposal_tasks_complete=sum(map(len,proposals)),
        rendered_batches_complete=len(rendered),rendered_tasks_complete=sum(map(len,rendered)),
        judged_batches_complete=len(labels),tasks_complete=sum(map(len,labels)),tasks_required=448,
        teacher_agent='crop-judge-terra',teacher_session='01a1b826-36c2-7922-a393-ed3007317e57')
    print(json.dumps(load(a.output/'progress.json')))


def render_many(a):
    import argparse
    for name in a.batch_indices.split(','):
        if len(name)!=2 or not name.isdigit():raise ValueError('two-digit explicit batch indices required')
        args=argparse.Namespace(**vars(a));args.batch=a.output/'batches'/f'{name}-sources.json';args.proposals=a.output/'batches'/f'{name}-proposals.json'
        render(args)
    status(a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','resolve','acquire','render','status','render_many']);p.add_argument('inputs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--batch-indices');p.add_argument('--variant-root',type=Path);p.add_argument('--manifests',type=Path);p.add_argument('--quarantine',type=Path);p.add_argument('--executable',type=Path);p.add_argument('--batch',type=Path);p.add_argument('--proposals',type=Path)
    a=p.parse_args()
    try:globals()[a.action](a)
    except Exception as e:
        if a.output.exists():progress(a.output,'failed',action=a.action,error=str(e))
        raise


if __name__=='__main__':main()
