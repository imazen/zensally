#!/usr/bin/env bash
set -euo pipefail
root=$1
export TMPDIR="${root}/scratch"
export PYTHONPATH="${root}/python-extra${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/opt/homebrew/bin:${HOME}/.cargo/bin:${PATH}"
mkdir -p "${TMPDIR}"
python3 - "${root}" <<'PY'
import hashlib,shutil,tarfile,urllib.request,sys
from pathlib import Path
r=Path(sys.argv[1]);archive=r/'ort-source.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!='ca15bee9ccf04f17da55d78834cce95168e3e2178edf402bd38d2e5eb95b1e26':raise ValueError('ORT source archive differs')
source=r/'onnxruntime-6af1a932e2286d5c58166085d24c68783033edee'
if not source.exists():
 with tarfile.open(archive) as t:t.extractall(r,filter='data')
stock=r/'ort-stock.tgz'
if not stock.exists():
 with urllib.request.urlopen('https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz') as inp,stock.open('xb') as out:shutil.copyfileobj(inp,out)
if not (r/'onnxruntime-osx-arm64-1.31.1').exists():
 with tarfile.open(stock) as t:t.extractall(r,filter='data')
print('stock archive SHA256',hashlib.sha256(stock.read_bytes()).hexdigest(),flush=True)
PY
python3 "${root}/tools/build.py" \
  "${root}/onnxruntime-6af1a932e2286d5c58166085d24c68783033edee" "${root}/build" \
  "${root}/yunet.ort" "${root}/yunet.required_operators_and_types.config" --jobs 3 \
  --stock-library "${root}/onnxruntime-osx-arm64-1.31.1/lib/libonnxruntime.1.31.1.dylib"
