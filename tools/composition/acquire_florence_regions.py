#!/usr/bin/env python3
"""Acquire the bounded native Florence scout with exact public asset identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import HfApi, snapshot_download

from artifact_identity import sha

REPO = 'florence-community/Florence-2-base'
REVISION = '00921df66db728a9ceb750f5eca43e5c203a2051'
FILES = ['README.md', 'config.json', 'generation_config.json', 'merges.txt',
         'model.safetensors', 'preprocessor_config.json', 'processor_config.json',
         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 4*1024**3:
        raise ValueError('four GiB disk headroom required')
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename in FILES]
    if (info.sha != REVISION or info.gated or info.private or info.card_data.get('license') != 'mit'
            or {f.rfilename for f in files} != set(FILES) or sum(f.size for f in files) > 600*1024**2):
        raise ValueError('registered public model contract differs')
    snapshot_download(REPO, revision=REVISION, allow_patterns=FILES, local_dir=args.output/'model',
                      max_workers=2, token=False)
    rows = []
    for file in files:
        path = args.output/'model'/file.rfilename
        digest = sha(path)
        if path.stat().st_size != file.size or (file.lfs and digest != file.lfs.sha256):
            raise ValueError('downloaded asset identity differs')
        rows.append(dict(path=str(path.relative_to(args.output)), bytes=file.size, sha256=digest,
                         url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{file.rfilename}'))
    license_url = 'https://huggingface.co/microsoft/Florence-2-base/resolve/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/LICENSE'
    with urllib.request.urlopen(license_url, timeout=30) as stream:
        license_bytes = stream.read(16385)
    if len(license_bytes) > 16384 or not license_bytes.startswith(b'MIT License'):
        raise ValueError('original license capture differs')
    (args.output/'MICROSOFT_LICENSE').write_bytes(license_bytes)
    rows.append(dict(path='MICROSOFT_LICENSE', bytes=len(license_bytes),
                     sha256=hashlib.sha256(license_bytes).hexdigest(), url=license_url))
    (args.output/'asset-identities.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION,
        license='mit', remote_code_permitted=False, files=rows), indent=2)+'\n')
    print(json.dumps(dict(repo=REPO, revision=REVISION, files=len(rows), bytes=sum(r['bytes'] for r in rows))))


if __name__ == '__main__':
    main()
