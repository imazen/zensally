#!/usr/bin/env python3
"""Replay the frozen real72 tile proposal packet through the shared mapper."""
import argparse
import json
from pathlib import Path
from artifact_identity import sha
from subject_tiled_faces import map_tile_faces


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if sha(a.root/'SHA256SUMS.json')!='c7872a992e607cb062089fdd678a0629a16d9f86841c831a4020cd9b8a579b29':raise ValueError('frozen tile packet differs')
    tiles={r['id']:r for r in json.loads((a.root/'manifest.json').read_text())['images']};rows=json.loads((a.root/'local/yunet/results.json').read_text());actual=[]
    for row in rows:
        tile=tiles[row['id']]
        actual.extend(dict(source_id=tile['source_id'],**r) for r in map_tile_faces(tile,row['result']['faces']))
    if actual!=json.loads((a.root/'analysis/tile-proposals.json').read_text()):raise ValueError('shared mapping changed frozen proposals')
    result=dict(views=len(rows),proposals=len(actual),exact=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
