"""Verified frozen inputs shared by person-risk analysis and numerical checks."""
import json
from pathlib import Path
import torch
from artifact_identity import sha
from audit_face_corroboration import verify_member
from distill_attention_linear import Student
from pair_relationship_head import PairCorrection
from person_risk_policy import person_count


def load(fits,miap,unsplash):
    if sha(fits/'SHA256SUMS.json')!='07c10b48620eb4badf95ca815f7d67bae636d0a32282270a660f05ba7d93f3de':raise ValueError('frozen classifier packet differs')
    for relative in ['routing/protocol.json','numerical/summary.json','summary.json']:verify_member(fits,relative)
    paths=json.loads((fits/'routing/protocol.json').read_text())['inputs'];paths.update(json.loads((fits/'numerical/summary.json').read_text())['inputs']);by_hash={v:Path(k) for k,v in paths.items()}
    for path,digest in paths.items():
        if sha(Path(path))!=digest:raise ValueError('frozen parent changed: '+path)
    new=torch.load(by_hash['a21209768e3d8fa547eef84bdf1699aff86f7e6abe2db5e2315436a36fbc22b3'],map_location='cpu',weights_only=False);old=torch.load(by_hash['41777134194b707c39d8362554afd4adb003e073ca1b31be6554390cbfd7a3af'],map_location='cpu',weights_only=False);rank=torch.load(by_hash['2583944f230150f675dfc9dcc94ee4e1cbb8d63e6b15aeb33eab02ee97bf6031'],map_location='cpu',weights_only=False);aux=torch.load(by_hash['848f2647921697f112f106cf13b11b016f24aa13b76b6621f5c77d4728f6eddc'],map_location='cpu',weights_only=False);ix=aux['indices']
    if aux['sources']!=[old['sources'][i] for i in ix] or old['sources']!=rank['sources'] or any(old['sources'][i]['role']!='dev' for i in ix):raise ValueError('original source alignment differs')
    original=dict(sources=aux['sources'],ratios=old['ratios'],hidden=old['hidden'][ix],eligible=old['eligible'][ix],query_boxes=rank['boxes'][ix].repeat_interleave(2,1),extras=aux['extras'],permuted_extras=aux['permuted_extras'],targets=old['targets'][ix],query_valid=old['query_valid'][ix]);s=torch.load(by_hash['4b801ad420688a167056ee6439d5c40c0f51fba31db46554f2c68113bf60bb3b'],map_location='cpu',weights_only=False)['model'];endpoint=Student({k:s['base.'+k] for k in ['weight','bias']},s['teacher'],s['mean'],s['std'],nonlinear=True);endpoint.load_state_dict(s);endpoint.eval();pair=PairCorrection('additive-linear',s['mean'],s['std']);pair.load_state_dict(torch.load(by_hash['05f41e086a3043bce8797b33b3e2d932c1117025c074101161a73b72aa52dede'],weights_only=False)['model'])
    coverage=by_hash['160d693d91bae8dd858036c0bd198088af26be847122e50bdad8fa2ec336a34a'].parent
    for relative in ['sources.json','queries.json']:verify_member(coverage,relative)
    head_rows={r['id']:r for r in json.loads((coverage/'sources.json').read_text())};queries={(r['id'],r['query']):r for r in json.loads((coverage/'queries.json').read_text())};contexts={};person_records=[]
    identities=dict(paths)
    for cohort,c,root,mhash,rhash in [('new',new,miap,'eb83a50e5c6760727b495d4d2e28b1f9d8c5fa23b1f844b3fc65c9beec0d1fa7','e5cf955399185392c7413aa8620ddb5113aa2f65a1bd6a31f75a3f40a6bbd3a9'),('old',original,unsplash,'69732eecab8057040a9668f8a6006a9d861d398732df1b034610e4f9cd2cb4a0','8e0caeab7b6d5a848c21fb520ba2bc11074900ee88b386279c2c21f2a7359407')]:
        mp=root/'manifest.json';rp=root/'full/tf-efficientdet_lite2/results.json'
        if sha(mp)!=mhash or sha(rp)!=rhash:raise ValueError('frozen person detector inputs differ')
        identities.update({str(mp):mhash,str(rp):rhash});manifest=json.loads(mp.read_text());images={r['id']:r for r in manifest['images']};objects={r['id']:r['result']['detections'] for r in json.loads(rp.read_text())['rows']}
        if manifest['confidence_threshold']!=.3 or manifest['max_results']!=50:raise ValueError('person detector contract differs')
        counts=[]
        for source in c['sources']:
            if source['role'] not in ['fit','dev']:raise ValueError('TEST stays closed')
            n=person_count(source,images[source['id']],objects[source['id']]);counts.append(n);person_records.append(dict(cohort=cohort,id=source['id'],role=source['role'],family_id=source['family_id'],people=n,applies=n>0,source=source['source']))
        count=len(counts);scale=torch.tensor([[*v['native_size'],*v['native_size']] for v in c['sources']])[:,None,None,:];native=(c['query_boxes']*scale).round().long().reshape(count,11,2,128,4);e=c['eligible'].reshape(count,11,2,128);o=native[:,:,0,:,None,:];inner=native[:,:,1,None,:,:];oa=(o[...,2]-o[...,0])*(o[...,3]-o[...,1]);ia=(inner[...,2]-inner[...,0])*(inner[...,3]-inner[...,1]);mask=(o[...,:2]<=inner[...,:2]).all(-1)&(o[...,2:]>=inner[...,2:]).all(-1)&(ia<oa)&e[...,0,:,None]&e[...,1,None,:]
        with torch.no_grad():h=c['hidden'].reshape(count,11,2,128,96);base=endpoint(h,None);scores=pair(h,base,e)
        if not mask.any((-2,-1)).all():raise ValueError('frozen pair feasibility changed')
        context=dict(data=c,native=native,scale=scale,eligible=e,mask=mask,scores=scores,person=torch.tensor(counts)>0)
        if cohort=='new':
            retained=torch.zeros_like(c['eligible']);qmeta={key:torch.zeros((count,22),dtype=torch.bool) for key in ['head_known','reference_known','contain','unknown']}
            for j,source in enumerate(c['sources']):
                row=head_rows[source['id']];heads=torch.tensor(row['head_natives']);nb=native[j].reshape(22,128,4)
                if len(heads):retained[j]=((nb[...,None,:2]<=heads[:,:2]).all(-1)&(nb[...,None,2:]>=heads[:,2:]).all(-1)).any(-1)&c['eligible'][j]
                for q in range(22):
                    meta=queries[source['id'],q];qmeta['head_known'][j,q]=meta['head_known'];qmeta['reference_known'][j,q]=meta['reference_known'];qmeta['contain'][j,q]=meta['recommendation']=='contain';qmeta['unknown'][j,q]=meta['recommendation']=='unknown'
            context.update(retained=retained,metadata=qmeta)
        contexts[cohort]=context
    fits_summary=json.loads((fits/'summary.json').read_text())['fits']
    if len(fits_summary)!=15:raise ValueError('all fifteen frozen heads required')
    return contexts,fits_summary,person_records,identities
