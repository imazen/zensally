#!/usr/bin/env python3
"""Run the Mac probe alone in a fresh directory and retain loader evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--wrapper', type=Path, required=True)
a = p.parse_args()
r = a.root.resolve()
isolated = r/'isolated'
isolated.mkdir()
exe = isolated/'ort-native-single'
shutil.copy2(r/'build/MinSizeRel'/exe.name, exe)
deps = subprocess.check_output(['otool', '-L', str(exe)], text=True)
(r/'dependencies.txt').write_text(deps)
for line in deps.splitlines()[1:]:
    name = line.strip().split(' (')[0]
    if not name.startswith(('/usr/lib/', '/System/Library/')):
        raise ValueError('non-system library: '+name)
env = dict(os.environ, PATH='/usr/bin:/bin:/usr/sbin:/sbin')
for key in list(env):
    if key.startswith('DYLD_'): del env[key]
result = subprocess.run([str(a.wrapper), '--mem', '2G', '--jobs', '1', '--', str(exe)],
                        cwd=isolated, env=env, check=True, capture_output=True, text=True)
(r/'standalone.stdout').write_text(result.stdout)
(r/'standalone.stderr').write_text(result.stderr)
if list(isolated.iterdir()) != [exe]: raise ValueError('unexpected companion file')
stock = r/'onnxruntime-osx-arm64-1.31.1/lib/libonnxruntime.1.31.1.dylib'
(r/'deployment.json').write_text(json.dumps(dict(platform=platform.platform(), machine=platform.machine(),
    executable_bytes=exe.stat().st_size, executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
    embedded_model_bytes=(r/'yunet.ort').stat().st_size, isolated_files=[exe.name],
    stock_library_bytes=stock.stat().st_size,
    stock_probe_bytes=(r/'build/MinSizeRel/ort-native-stock').stat().st_size), indent=2)+'\n')
print(result.stdout)
print(result.stderr)
