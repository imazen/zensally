#!/usr/bin/env python3
"""Persist available model IDs only, excluding request credentials."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from urllib.request import Request,urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    request=Request('https://api.openai.com/v1/models',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
    with urlopen(request,timeout=30) as reply:data=json.loads(reply.read(2*1024*1024))
    wanted=['gpt-image-2.6-sunburst','gpt-6.1-terra','gpt-6.1-sol']
    models=sorted(r['id'] for r in data['data'])
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),models=models,requested_available={m:m in models for m in wanted},paid_requests=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(requested_available=result['requested_available'],paid_requests=0)))


if __name__=='__main__':main()
