#!/usr/bin/env python3
"""Run a child with an authorized SSH-host API credential, without logging it."""
import argparse
import os
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('host');p.add_argument('env_file');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not a.command:raise ValueError('child command required')
    remote='''import json,shlex,sys
from pathlib import Path
path=Path(sys.argv[1]).expanduser();found=[]
for line in path.read_text().splitlines():
 try:tokens=shlex.split(line,comments=True)
 except ValueError:continue
 for token in tokens:
  if token.startswith('OPENAI_API_KEY='):found.append(token.split('=',1)[1])
if len(found)!=1 or not found[0]:raise SystemExit('one configured API credential required')
sys.stdout.write(found[0])
'''
    # Both arguments are positional SSH command arguments, never shell-expanded
    # credential text. Only the key travels in the captured pipe, never stdout.
    import shlex
    command='python3 - '+shlex.quote(a.env_file)
    key=subprocess.check_output(['ssh',a.host,command],input=remote.encode()).decode().strip()
    if not key or '\n' in key:raise ValueError('invalid credential transport')
    environment={**os.environ,'OPENAI_API_KEY':key}
    os.execvpe(a.command[0],a.command,environment)


if __name__=='__main__':main()
