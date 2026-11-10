#!/usr/bin/env python3
"""Inventory unused TRAIN people/product metadata without opening source pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crop-first'))
from corpus import reconcile,rows
from artifact_identity import sha


def author(value):
    return value.strip().casefold() if isinstance(value,str) else ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','registry','output']:
        p.add_argument(name,type=Path)
    args = p.parse_args()
    if sha(args.registry) != '6f8d318cc870248dcc2400a235fcbaa5f7f0edf47d575d965bf5c72b5684d2f6':
        raise ValueError('frozen photographic source roles differ')
    protocol = json.loads((args.corpus/'protocol.json').read_text())
    for link in protocol['links']:
        if sha(args.corpus/link['text']) != link['sha256']:
            raise ValueError('pinned corpus metadata differs')
    records = reconcile({role:rows((args.corpus/f'jxl-{role}.tsv').read_bytes())
        for role in ['train','validate','test']},rows((args.corpus/'family-v2.tsv').read_bytes()),
        sum([rows((args.corpus/f'historical-{role}.tsv').read_bytes()) for role in ['train','validate','test']],[]))
    selected = json.loads((args.corpus/'sources.json').read_text())
    ids,families = {s['id'] for s in selected},{s['family_id'] for s in selected}
    keys = ['original_sha256','pixels_sha256','sha256']
    hashes = {(k,s[k]) for s in selected for k in keys if s.get(k)}
    photographic = json.loads(args.registry.read_text())['sources']
    authors,provider_ids = {},{}
    for source in photographic:
        a = author(source['source']['photographer'])
        if a:
            authors.setdefault(a,[]).append(dict(namespace='photographic-registry',id=source['id'],role=source['role']))
        provider_ids[source['source']['provider_id'].casefold()] = source
    for source in selected:
        a = author(source['provenance'].get('photographer',''))
        if source['source']=='unsplash' and a:
            authors.setdefault(a,[]).append(dict(namespace='prior-current-jxl-selection',id=source['id'],role=source['use']))
    candidates = []
    for source in records:
        kind = ('people' if source['source']=='unsplash' and source['category']=='people' else
                'synthetic_product' if source['source']=='lilith-ai' and source['category']=='products' else None)
        if (kind is None or source['bucket']!='train' or source['historical_eval_family']
                or source['id'] in ids or source['family_id'] in families
                or any((k,source[k]) in hashes for k in keys if source.get(k))):
            continue
        p = source['provenance']
        matches = authors.get(author(p.get('photographer','')),[]) if kind=='people' else []
        url = p.get('source_url','')
        provider_match = next((s for identifier,s in provider_ids.items()
            if kind=='people' and urlparse(url).path.rstrip('/').casefold().endswith(identifier)),None)
        exact_original = [s['id'] for s in photographic
            if source.get('original_sha256') and source['original_sha256']==s['source']['sha256']]
        roles = {m['role'] for m in matches}
        status = ('existing_provider_or_original_alias' if provider_match or exact_original else
                  'quarantine_prior_test_or_conflicting_author_roles' if roles & {'eval','test'} or len(roles)>1 else
                  'author_barrier_'+next(iter(roles)) if roles else 'unassigned_metadata_candidate')
        candidates.append(dict(source,candidate_kind=kind,original_canonical_bucket='train',
            prior_author_matches=matches,provider_alias=provider_match['id'] if provider_match else None,
            exact_original_matches=exact_original,metadata_status=status,
            synthetic=kind=='synthetic_product',new_role=None,training_eligible=False,
            pixels_opened_for_this_inventory=False,visual_review_complete=False))
    counts = Counter(r['candidate_kind'] for r in candidates)
    if counts != Counter(people=10,synthetic_product=362):
        raise ValueError('registered unused TRAIN metadata inventory differs')
    args.output.mkdir(parents=True,exist_ok=False)
    result = dict(candidates=candidates,build_commit=subprocess.check_output(
        ['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
        corpus_protocol_sha256=sha(args.corpus/'protocol.json'),prior_selection_sha256=sha(args.corpus/'sources.json'),
        photographic_registry_sha256=sha(args.registry),links=protocol['links'],
        canonical_holdout_pixels_opened=False,new_roles_assigned=False,prior_roles_changed=False,
        new_training_labels=False,paid_model_api_calls=0)
    with (args.output/'inventory.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    (args.output/'links.jsonl').write_text(''.join(json.dumps(link)+'\n' for link in protocol['links']))
    summary = dict(groups={kind:dict(sources=sum(r['candidate_kind']==kind for r in candidates),
        families=len({r['family_id'] for r in candidates if r['candidate_kind']==kind}),
        metadata_status=dict(Counter(r['metadata_status'] for r in candidates if r['candidate_kind']==kind)),
        encoded_bytes=sum(int(r['bytes']) for r in candidates if r['candidate_kind']==kind),
        largest_native_pixels=max(int(r['width'])*int(r['height']) for r in candidates if r['candidate_kind']==kind))
        for kind in counts},people_candidates=[dict(id=r['id'],family_id=r['family_id'],
        metadata_status=r['metadata_status'],prior_author_matches=r['prior_author_matches'],provider_alias=r['provider_alias'])
        for r in candidates if r['candidate_kind']=='people'],
        inventory_sha256=sha(args.output/'inventory.json'),new_roles_assigned=False,pixels_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
