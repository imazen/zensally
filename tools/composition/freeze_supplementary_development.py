#!/usr/bin/env python3
"""Reserve source-bound development diagnostics without moving prior roles."""
import argparse
import json
from pathlib import Path
import subprocess

from artifact_identity import sha

OWNED_IDS = ['11046','11048','11054','11064','11066','11214','11440']
NOTES = {
    '12002':'Single environmental portrait with hat and hand gesture; looking room to image right.',
    '12006':'Close tilted portrait with sunglasses and hand-to-ear interaction.',
    '12010':'Overhead reader among shelves; retain book/hand relationship.',
    '12012':'Two foreground people leaning through vehicle window; both heads and hands matter.',
    '12014':'Three widely separated foreground people at beach with city backdrop; mixed orientations.',
    '12016':'Extreme close profile; original source-edge truncation does not permit further face loss.',
    '12022':'Market interaction with foreground person and partial body at left source edge.',
    '12026':'Seated environmental portrait in landscape, facing image left.',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['people','owned','owned_review','oracles','relationships','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    if sha(args.people/'sources.json')!='5e125f64119e4d8c2a8044fc8a21154d79e5c374f0621938aad6543456c4d1c8':
        raise ValueError('registered acquired people metadata differs')
    people = json.loads((args.people/'sources.json').read_text())
    owned = {s['id']:s for s in json.loads((args.owned/'sources.json').read_text())}
    review = {s['id']:s for s in json.loads((args.owned_review/'content-review.json').read_text())['observations']}
    canonical = json.loads((args.oracles/'canonical-pixels.json').read_text())
    png = json.loads((args.oracles/'png-tensor-oracle.json').read_text())
    relationships = json.loads((args.relationships/'summary.json').read_text())
    if (canonical['exact_sources']!=8 or len(png['checks'])!=8
            or not all(all(r[k] for k in ['source_png_rgb_exact','preview_png_rgb_exact','tensor_hash_and_padding']) for r in png['checks'])
            or relationships['exact_conflicts'] or relationships['pairs']!=21582):
        raise ValueError('complete source or relationship checks required')
    flagged = {r[k] for r in relationships['possible_relationships'] for k in ['left','right']}
    sources = []
    for source in people+[owned[k] for k in OWNED_IDS]:
        identifier = 'imazen26:'+source['id']
        if (source['bucket']!='train' or source['historical_eval_family'] or source['use']!='screening'
                or source['training_eligible'] or identifier in flagged):
            raise ValueError('candidate has prior eligibility, holdout or relationship conflict')
        raw = Path(source['directory'])/'preview.rgb'
        w,h = int(source['width']),int(source['height'])
        pw,ph = max(1,round(w*768/max(w,h))),max(1,round(h*768/max(w,h)))
        if raw.stat().st_size!=pw*ph*3 or sha(Path(source['directory'])/'preview.png')!=source['preview_sha256']:
            raise ValueError('actual source preview differs')
        note = NOTES.get(source['id']) or review[identifier]['note']
        sources.append(dict(id=identifier,family_id='imazen26:'+source['family_id'],
            role='supplementary_dev',canonical_bucket='train',prior_use='screening',fit_eligible=False,
            rgb=str(raw),rgb_sha256=sha(raw),native_rgb_sha256=source['rgb_sha256'],
            native_size=[w,h],working_size=[pw,ph],preview_sha256=source['preview_sha256'],
            tensor_path=str(Path(source['directory'])/'tensor.rgb'),tensor_sha256=source['tensor_sha256'],
            valid_rect=source['valid_rect'],source=dict(provider_id=source['id'],namespace='imazen26-current-jxl',
                source_url=source['canonical_url'],original_provider_url=source['provenance'].get('source_url'),
                photographer=source['provenance'].get('photographer')),
            query='The monkeys.' if source['id']=='11440' else 'The people.',
            content_note=note,original_source_record=source,human_rated=False,
            face_coverage_certified=False,training_labels_admitted=False))
    if len(sources)!=15 or len({s['family_id'] for s in sources})!=15:
        raise ValueError('exact fifteen distinct canonical source families required')
    args.output.mkdir(parents=True,exist_ok=False)
    result = dict(schema='supplementary-crop-development-v1',sources=sources,
        build_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        bindings={key:sha(path) for key,path in [('people',args.people/'sources.json'),('owned',args.owned/'sources.json'),
            ('owned_review',args.owned_review/'content-review.json'),('native_pixels',args.oracles/'canonical-pixels.json'),
            ('png_tensor',args.oracles/'png-tensor-oracle.json'),('relationships',args.relationships/'summary.json')]},
        new_development_sources=15,prior_role_assignments_changed=False,original_canonical_buckets_changed=False,
        fitting_sources=0,labels_admitted=False,test_sources=0,unseen_confirmation_claimed=False,
        paid_model_api_calls=0,face_coverage_certified=False)
    path = args.output/'sources.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(development_sources=15,families=15,fitting_sources=0,labels_admitted=False,
                         source_manifest_sha256=sha(path))))


if __name__ == '__main__':
    main()
