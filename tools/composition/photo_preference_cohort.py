#!/usr/bin/env python3
"""Select the existing photographic screening categories without moving roles."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'crop-first'))
from photo_focus import category_scope
from artifact_identity import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'rendered', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    members, packets = [], []
    for index, (source, use) in enumerate(zip(cache['sources'], cache['uses'])):
        path = args.rendered/source['id']/'packet.json'
        packet = json.loads(path.read_text())
        if any(source[key] != packet[key] for key in ['id', 'use', 'family_id', 'sha256']):
            raise ValueError('rendered/source identity differs')
        category = packet['provenance']['category']
        if category_scope(category) == 'photographic_screening_candidate':
            members.append(dict(index=index, id=source['id'], family_id=source['family_id'],
                                use=use, category=category))
            packets.append(dict(path=str(path), sha256=sha(path)))
    counts = {role: sum(r['use'] == role for r in members) for role in ['fit', 'dev']}
    if counts != dict(fit=15, dev=5):
        raise ValueError('registered photographic cohort differs')
    data = dict(schema=1, name='existing-photo-preferences-v1', cache_sha256=sha(args.cache),
                members=members, counts=counts, packets=packets,
                selector='existing people-commerce-v1 metadata category scope; not a visual-content certification',
                roles_moved=False, new_labels_admitted=False, canonical_holdouts_opened=False)
    with args.output.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
