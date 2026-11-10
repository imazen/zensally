#!/usr/bin/env python3
"""Screen acquired TRAIN candidates against frozen source identities and fingerprints."""
import argparse
import atexit
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from artifact_identity import sha
from audit_unsplash_relationships import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['people','owned','old_selection','synthetic_admission','photo_fingerprints','binary','resizer','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    for path,digest in [(args.people/'sources.json','5e125f64119e4d8c2a8044fc8a21154d79e5c374f0621938aad6543456c4d1c8'),
        (args.owned/'sources.json','175162d764bfe098176e8c29f6e5a6bfe8f4d0e3ec31363e89013e9950fb807a'),
        (args.photo_fingerprints,'e7220dd1db356ea3c1b00db864dfaf7baa896130a303a38e717fc678068f828d'),
        (args.binary,'3113e25a9ac4f037eac705445c598223771796491ad6b266889445abaeb0aef0'),
        (args.resizer,'ada0b5f65204ed1be4ed93da6fd244d214c47bf11a5852f2d3ebe6625b003041')]:
        if sha(path)!=digest:
            raise ValueError('registered source or owned fingerprint/resizer differs')
    people = json.loads((args.people/'sources.json').read_text())
    owned = json.loads((args.owned/'sources.json').read_text())
    old = json.loads(args.old_selection.read_text())
    synthetic = json.loads(args.synthetic_admission.read_text())['sources']
    photo = json.loads(args.photo_fingerprints.read_text())
    if (len(people),len(owned),len(old),len(synthetic),len(photo)) != (8,101,104,56,144):
        raise ValueError('registered source-reference counts differ')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,args.output/'corpus-fingerprint')
    (args.output/'calibration').mkdir()
    status = dict(pid=os.getpid(),stage='fingerprinting',completed=0,required=109,checked_unix=time.time())
    progress = args.output/'progress.json'
    def final_status():
        if status['pid']==os.getpid():
            status.update(pid=None,stage='failed-or-interrupted',checked_unix=time.time())
            progress.write_text(json.dumps(status)+'\n')
    atexit.register(final_status)
    progress.write_text(json.dumps(status)+'\n')
    hashes,calibration,exact_conflicts = [],[],[]
    for source in owned+people:
        if source['bucket']!='train' or source['use']!='screening' or source['training_eligible'] or source['historical_eval_family']:
            raise ValueError('only unadmitted non-holdout TRAIN sources permitted')
        width,height = int(source['width']),int(source['height'])
        pw,ph = max(1,round(width*768/max(width,height))),max(1,round(height*768/max(width,height)))
        rgb = Path(source['directory'])/'preview.rgb'
        if rgb.stat().st_size!=pw*ph*3:
            raise ValueError('actual owned preview extent differs')
        value = fingerprint(args.binary,rgb,pw,ph,pw*3)
        row = dict(id='imazen26:'+source['id'],family_id='imazen26:'+source['family_id'],
            hash=value['hash'],working_rgb_sha256=sha(rgb),native_rgb_sha256=source['rgb_sha256'],
            canonical_pixels_sha256=source['pixels_sha256'],encoded_sha256=source['sha256'],
            original_sha256=source.get('original_sha256'),source_url=source['canonical_url'])
        for reference in old:
            reasons = [key for key in ['id','family_id','original_sha256','pixels_sha256','sha256']
                if source.get(key) and source.get(key)==reference.get(key)]
            if reasons:
                exact_conflicts.append(dict(id=row['id'],reference='old-current-jxl:'+reference['id'],reasons=reasons))
        for reference in synthetic:
            if row['native_rgb_sha256']==reference['rgb_sha256']:
                exact_conflicts.append(dict(id=row['id'],reference='synthetic:'+reference['id'],reasons=['native_rgb_sha256']))
        if source in people:
            tw,th = max(1,round(width*320/max(width,height))),max(1,round(height*320/max(width,height)))
            path = args.output/'calibration'/(source['id']+'-320.rgb')
            with rgb.open('rb') as stream:
                small = subprocess.check_output([str(args.resizer),str(pw),str(ph),str(pw*3),
                    f'0,0,{pw},{ph},{tw},{th}'],stdin=stream)
            if len(small)!=tw*th*3:
                raise ValueError('calibration resize extent differs')
            path.write_bytes(small)
            other = fingerprint(args.binary,path,tw,th,tw*3)
            calibration.append(dict(id=row['id'],preview768_hash=row['hash'],preview320_hash=other['hash'],
                hamming_distance=(int(row['hash'],16)^int(other['hash'],16)).bit_count(),rgb_sha256=sha(path)))
        hashes.append(row)
        status.update(completed=len(hashes),source_id=row['id'],checked_unix=time.time())
        progress.write_text(json.dumps(status)+'\n')
    pairs = []
    for left in hashes:
        for right in photo:
            pairs.append(dict(left=left['id'],right=right['id'],kind='supplementary_vs_photographic',
                hamming_distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count()))
    for left,right in combinations(hashes,2):
        pairs.append(dict(left=left['id'],right=right['id'],kind='supplementary_internal',
            same_canonical_family=left['family_id']==right['family_id'],
            hamming_distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count()))
    if len(pairs)!=21582 or sum(r['kind']=='supplementary_vs_photographic' for r in pairs)!=15696:
        raise ValueError('complete pair matrix differs')
    flags = [r for r in pairs if r['hamming_distance']<=10]
    result = dict(sources=109,photo_reference_sources=144,pairs=len(pairs),possible_relationships=flags,
        exact_conflicts=exact_conflicts,threshold=10,calibration=calibration,
        binary_sha256=sha(args.binary),resizer_sha256=sha(args.resizer),
        bindings={name:sha(path) for name,path in [('people',args.people/'sources.json'),('owned',args.owned/'sources.json'),
            ('old_selection',args.old_selection),('synthetic_admission',args.synthetic_admission),('photo_fingerprints',args.photo_fingerprints)]},
        new_roles_assigned=False,training_labels_admitted=False,complete_duplicate_recall_certified=False,
        canonical_holdout_pixels_opened=False,paid_model_api_calls=0)
    for name,value in [('fingerprints.json',hashes),('pair-distances.json',pairs),('summary.json',result)]:
        with (args.output/name).open('x') as stream:
            stream.write(json.dumps(value,indent=2)+'\n')
    status.update(pid=None,stage='complete',checked_unix=time.time())
    progress.write_text(json.dumps(status)+'\n')
    print(json.dumps(dict(sources=109,pairs=len(pairs),possible_relationships=len(flags),exact_conflicts=len(exact_conflicts),
        cross_corpus_flags=sum(r['kind']=='supplementary_vs_photographic' for r in flags),
        calibration_distances=[r['hamming_distance'] for r in calibration])))
    if exact_conflicts:
        raise ValueError('source identity conflict; retained complete comparison before any admission')


if __name__ == '__main__':
    main()
