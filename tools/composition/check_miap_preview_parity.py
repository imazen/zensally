#!/usr/bin/env python3
"""Require fresh shared-helper outputs to match all twelve frozen witness pixels."""
import argparse
import json
from pathlib import Path
import shutil

from artifact_identity import sha
from miap_owned_preview import prepare_preview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['witnesses','pipeline','reader','output']:p.add_argument(name,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.witnesses/'protocol.json').read_text());rows=[]
    for identifier in protocol['selected_ids']:
        key=identifier.split(':')[1];old=a.witnesses/key;directory=a.output/key;directory.mkdir()
        row=json.loads((old/'record.json').read_text());shutil.copy2(old/'source.jpg',directory/'source.jpg')
        actual=prepare_preview(directory,a.pipeline,a.reader,row['source'],row['download'],row['boxes'])
        if actual!=row:raise ValueError('shared helper changes source/geometry/readback contract')
        if any(sha(directory/name)!=sha(old/name) for name in ['preview.png','preview.rgb']):raise ValueError('shared helper changes encoded/native preview bits')
        rows.append(dict(id=identifier,png_sha256=sha(directory/'preview.png'),rgb_sha256=sha(directory/'preview.rgb'),exact=True))
    result=dict(sources=len(rows),exact_sources=sum(r['exact'] for r in rows),checks=rows,training_eligible=False,paid_model_api_calls=0)
    (a.output/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(sources=len(rows),exact_sources=result['exact_sources'])))


if __name__=='__main__':main()
