#!/usr/bin/env python3
"""Copy clean owned build dependencies into a bounded immutable experiment input."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zenanalyze',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    def revision():
        subprocess.run(['git','-C',str(args.zenanalyze),'diff','--quiet'],check=True)
        subprocess.run(['git','-C',str(args.zenanalyze),'diff','--cached','--quiet'],check=True)
        return subprocess.check_output(['git','-C',str(args.zenanalyze),'rev-parse','HEAD'],text=True).strip()
    before=revision()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for crate in ['zenpredict','zenpredict-bake','zenanalyze-api']:
        source=args.zenanalyze/crate
        chosen=[f for f in source.rglob('*') if f.is_file() and
                (f.relative_to(source).parts[0] in ['src','docs'] or
                 f.name in ['Cargo.toml','README.md','README.crates.md','LICENSE','LICENSE-MIT','LICENSE-APACHE']) and
                f.name not in ['CLAUDE.md','AGENTS.md']]
        if len(chosen)>1000 or sum(f.stat().st_size for f in chosen)>32*1024**2:
            raise ValueError('owned source snapshot exceeds declared bound')
        for file in chosen:
            relative=file.relative_to(args.zenanalyze);target=args.output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            identity=digest(file);shutil.copy2(file,target)
            if digest(target)!=identity or digest(file)!=identity:raise ValueError('source changed during snapshot')
            rows.append(dict(path=str(relative),sha256=identity))
    if revision()!=before:raise ValueError('source revision changed during snapshot')
    (args.output/'_MANIFEST.json').write_text(json.dumps(dict(build_commit=before,files=rows,
        scope='owned bake/runtime/API sources for an unpublished build-time dependency; original repo unchanged'),indent=2)+'\n')
    print(json.dumps(dict(build_commit=before,files=len(rows))))


if __name__=='__main__':main()
