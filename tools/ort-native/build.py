#!/usr/bin/env python3
"""Build native reduced ORT with an embedded YuNet model and static runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('build', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('config', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--stock-library', type=Path)
    a = p.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    model = a.model.read_bytes()
    header = a.build/'embedded_model.h'
    header.write_text('alignas(64) static const unsigned char kModel[] = {\n'+
                      ',\n'.join(','.join(str(v) for v in model[i:i+32]) for i in range(0,len(model),32))+'\n};\n')
    tool = Path(__file__).resolve().parent
    defines = ['CMAKE_PROJECT_onnxruntime_INCLUDE='+str(tool/'probe.cmake'),
               'ZENSALLY_EMBED_DIR='+str(a.build.resolve()),
               'FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER']
    if sys.platform == 'darwin': defines.append('CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew')
    if os.name == 'nt':
        # The runner's default patch tool failed ORT's bundled ONNX patch.
        patch = Path(shutil.which('git')).resolve().parent.parent/'usr/bin/patch.exe'
        if not patch.is_file(): raise FileNotFoundError(patch)
        subprocess.run([str(patch), '--version'], check=True)
        defines.append('Patch_EXECUTABLE='+patch.as_posix())
    if os.name != 'nt': defines.append('CMAKE_CXX_FLAGS=-Wno-error=pass-failed')
    if a.stock_library: defines.append('ZENSALLY_STOCK_LIBRARY='+str(a.stock_library.resolve()))
    cmd = [sys.executable, str(a.source/'tools/ci_build/build.py'), '--build_dir', str(a.build),
           '--config', 'MinSizeRel', '--cmake_generator', 'Ninja', '--update', '--build',
           '--skip_tests', '--skip_submodule_sync', '--parallel', str(a.jobs),
           '--minimal_build', '--include_ops_by_config', str(a.config),
           '--enable_reduced_operator_type_support', '--disable_ml_ops', '--disable_rtti', '--no_telemetry',
           '--cmake_extra_defines', *defines]
    if os.name == 'nt':
        cmd.insert(-len(defines)-1, '--enable_msvc_static_runtime')
        # The Windows ARM assembler rejects KleidiAI's /arch:armv8.2 flag.
        # Retain the ORT CPU kernels and the unchanged output comparison gate.
        cmd.insert(-len(defines)-1, '--no_kleidiai')
    (a.build/'native-build-identity.json').write_text(json.dumps(dict(command=cmd,
        model_sha256=hashlib.sha256(model).hexdigest(), model_bytes=len(model)),indent=2)+'\n')
    subprocess.run(cmd,check=True)


if __name__ == '__main__': main()
