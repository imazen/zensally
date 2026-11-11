#!/usr/bin/env python3
"""Pinned Windows source bootstrap and real single-executable deployment check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

SOURCE = '6af1a932e2286d5c58166085d24c68783033edee'
SHA = {'source': 'ca15bee9ccf04f17da55d78834cce95168e3e2178edf402bd38d2e5eb95b1e26',
       'x64': '5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c',
       'arm64': '817f4129073f84f81cde3ff92b1a4e06d007691e7208deed08a8b6d6df143f7f'}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', choices=['build', 'prove'])
p.add_argument('root', type=Path)
p.add_argument('--arch', required=True, choices=['x64', 'arm64'])
a = p.parse_args()
r = a.root.resolve()
r.mkdir(parents=True, exist_ok=True)
tool = Path(__file__).resolve().parent
stock = r/f'onnxruntime-win-{a.arch}-1.31.1'
build = r/'build'/'MinSizeRel'

if a.stage == 'build':
    for key, url, dest in [
        ('source', f'https://codeload.github.com/microsoft/onnxruntime/tar.gz/{SOURCE}', r/'source.tgz'),
        (a.arch, f'https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/{stock.name}.zip', r/'stock.zip')]:
        with urllib.request.urlopen(url) as inp, dest.open('xb') as out:
            shutil.copyfileobj(inp, out)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != SHA[key]:
            raise ValueError('archive checksum mismatch: '+key)
    with tarfile.open(r/'source.tgz') as t:
        t.extractall(r/'source', filter='data')
    with zipfile.ZipFile(r/'stock.zip') as z:
        z.extractall(r)
    source = next((r/'source').iterdir())
    subprocess.run([sys.executable, str(tool/'build.py'), str(source), str(r/'build'),
                    str(r/'model'/'yunet.ort'), str(r/'model'/'yunet.required_operators_and_types.config'),
                    '--jobs', '2', '--stock-library', str(stock/'lib'/'onnxruntime.lib')], check=True)
    for dll in (stock/'lib').glob('*.dll'):
        shutil.copy2(dll, build/dll.name)
else:
    isolated = r/'isolated'
    isolated.mkdir()
    exe = isolated/'ort-native-single.exe'
    shutil.copy2(build/exe.name, exe)
    imports = subprocess.check_output(['dumpbin', '/DEPENDENTS', str(exe)], text=True)
    (r/'dependencies.txt').write_text(imports)
    # Reject ORT and dynamically linked compiler runtimes. Windows system DLLs remain allowed.
    lower = imports.lower()
    if any(name in lower for name in ['onnxruntime', 'vcruntime', 'msvcp', 'ucrtbase.dll']):
        raise ValueError('unexpected non-system deployment dependency')
    env = dict(os.environ, PATH=os.path.join(os.environ['SystemRoot'], 'System32'))
    result = subprocess.run([str(exe)], cwd=isolated, env=env, check=True, capture_output=True, text=True)
    (r/'standalone.stdout').write_text(result.stdout)
    if list(isolated.iterdir()) != [exe]:
        raise ValueError('deployment directory contains companion files')
    (r/'deployment.json').write_text(json.dumps(dict(platform=platform.platform(), machine=platform.machine(),
        executable_bytes=exe.stat().st_size, executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        embedded_model_bytes=(r/'model'/'yunet.ort').stat().st_size,
        isolated_files=[exe.name], stdout=result.stdout), indent=2)+'\n')
    print(result.stdout)
