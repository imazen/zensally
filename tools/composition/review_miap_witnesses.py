#!/usr/bin/env python3
"""Preserve bounded visual observations and compare owned stored fingerprints."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from artifact_identity import sha
from audit_unsplash_relationships import fingerprint

OBSERVATIONS={
    '75956e7d53852d58':('photo','Near portrait at water; person region aligned, source truncates lower body.'),
    'd0eebdf138acb5c5':('photo','Person painting a wall while standing on a chair; region aligned around person, excludes chair.'),
    '2cdeb68e6c417098':('photo','Small distant person near architectural entrance; region aligned.'),
    '10cafdec9a42e709':('group_count','Crowded event, one group region plus one individual; two box records do not mean two people.'),
    'c0bda1fe5ac25233':('photo','Two foreground people interacting at an event; both person regions align.'),
    '9028c1a74db2b386':('photo','Two distant people on watercraft; boxes align around seated people.'),
    '88612df00d855c68':('photo','Foreground person holding an object, plus background people and partial edge body; regions broadly align.'),
    '7df0a4df4df67865':('exclude_document','Magazine cover with printed portraits; outside photographic successor scope despite depiction flags0.'),
    '46953247dc4e523a':('quarantine_depiction_target','Outdoor photo includes real people and a mounted statue. One person box labels statue with depiction flag0; per-box review needed.'),
    '4d88936c4e12313b':('photo','Foreground interaction in an exhibition with background people; regions broadly align.'),
    'bf4eed789d9e2b5e':('photo','Street pedestrians at mixed scales, with vehicle occlusion; regions broadly align.'),
    '76a1663ff02cdcd1':('photo','Distant people along market stalls; regions broadly align at small scale.'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','binary','photographic','supplementary']:p.add_argument(name,type=Path)
    a=p.parse_args()
    for path,digest in [(a.binary,'3113e25a9ac4f037eac705445c598223771796491ad6b266889445abaeb0aef0'),
        (a.photographic,'e7220dd1db356ea3c1b00db864dfaf7baa896130a303a38e717fc678068f828d'),
        (a.supplementary,'0cb5e4abba51cec990062e2a18032723c1ed518671d6a4b38405376cab97d827')]:
        if sha(path)!=digest:raise ValueError('frozen fingerprint reference or owned binary differs')
    browser=json.loads((a.root/'browser-miap-witness.json').read_text())
    if browser['decoded']!=12 or browser['boxes']!=46 or browser['exceptions'] or browser['screenshot_bytes']>8*1024**2:raise ValueError('complete bounded browser evidence required')
    protocol=json.loads((a.root/'protocol.json').read_text());references=json.loads(a.photographic.read_text())+json.loads(a.supplementary.read_text())
    if len(references)!=253:raise ValueError('complete existing fingerprint inventory required')
    hashes=[];observations=[];exact=[]
    for identifier in protocol['selected_ids']:
        key=identifier.split(':')[1];row=json.loads((a.root/key/'record.json').read_text());path=a.root/key/'preview.rgb'
        if sha(path)!=row['preview.rgb_sha256']:raise ValueError('owned witness pixels differ')
        geometry=row['geometry'];value=fingerprint(a.binary,path,geometry['width'],geometry['height'],geometry['width']*3)
        hashes.append(dict(id=identifier,**value,rgb_sha256=sha(path)))
        status,note=OBSERVATIONS[key]
        observations.append(dict(id=identifier,status=status,note=note,reviewer='assistant visual inspection',
            human_annotation=False,face_completeness_certified=False,whole_body_completeness_certified=False,
            training_eligible=False,source_role='fit',source_role_changed=False))
        for reference in references:
            if row['download']['sha256'] in {reference.get('original_sha256'),reference.get('encoded_sha256')} or sha(path) in {reference.get('rgb_sha256'),reference.get('working_rgb_sha256')}:
                exact.append(dict(id=identifier,reference=reference['id']))
    pairs=[dict(left=left['id'],right=right['id'],kind='new_vs_existing',distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count()) for left in hashes for right in references]
    pairs += [dict(left=left['id'],right=right['id'],kind='new_internal',distance=(int(left['hash'],16)^int(right['hash'],16)).bit_count()) for left,right in combinations(hashes,2)]
    if len(pairs)!=3102:raise ValueError('complete available-reference pair coverage required')
    report=dict(fingerprint_binary_sha256=sha(a.binary),references=253,pairs=len(pairs),existing_reference_pairs=3036,
        threshold=10,possible_relationships=[r for r in pairs if r['distance']<=10],exact_matches=exact,
        fingerprints=hashes,observations=observations,
        out_of_scope_document_sources=1,depiction_target_quarantines=1,group_count_witnesses=1,
        reference_bindings={str(p):sha(p) for p in [a.photographic,a.supplementary]},
        complete_duplicate_recall_certified=False,canonical_holdout_pixels_opened=False,
        training_eligible=False,labels_admitted=False,face_coverage_certified=False,paid_model_api_calls=0)
    for name,value in [('review-and-relationships.json',report),('pair-distances.json',pairs)]:
        with (a.root/name).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['references','pairs','possible_relationships','exact_matches','out_of_scope_document_sources','depiction_target_quarantines','group_count_witnesses']}))


if __name__=='__main__':main()
