#!/usr/bin/env python3
"""Measure stripped example bytes and cold-process max RSS; no timing claims."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('this probe requires Linux /usr/bin/time -v and GNU strip')
    for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']:
        if 'target-cpu=native' in os.environ.get(key, ''):
            raise RuntimeError('runtime-dispatch comparison must not use target-cpu=native')
    strip = shutil.which('strip')
    if not strip:
        raise RuntimeError('GNU strip is required')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name in ['micro', 'tract']:
        source = args.examples / ('weight_' + name)
        binary = args.output / name
        shutil.copy2(source, binary)
        subprocess.run([strip, '--strip-all', str(binary)], check=True)
        memory_log = args.output / (name + '.time-v.txt')
        env = os.environ.copy()
        env['LC_ALL'] = 'C'
        with (args.output / (name + '.stdout.txt')).open('xb') as stdout, \
             (args.output / (name + '.stderr.txt')).open('xb') as stderr:
            subprocess.run(['/usr/bin/time', '-v', '-o', str(memory_log), str(binary)],
                           stdout=stdout, stderr=stderr, env=env, check=True)
        checksum = float((args.output / (name + '.stdout.txt')).read_text().strip())
        if not math.isfinite(checksum):
            raise RuntimeError('nonfinite model output')
        match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', memory_log.read_text())
        if not match:
            raise RuntimeError('max RSS missing from /usr/bin/time output')
        rows[name] = {'stripped_bytes': binary.stat().st_size,
                      'stripped_sha256': sha256(binary), 'source_executable_sha256': sha256(source),
                      'cold_process_max_rss_kib': int(match[1]), 'output_sum': checksum}
    summary = {'source_commit': args.source_commit, 'schema_version': 1,
               'scope': 'one fresh process per arm: load model and complete first 256x256 inference; resize excluded',
               'micro_mode': 'fast sigmoid; std; no AVX512 feature',
               'compiler': subprocess.check_output(['rustc', '-Vv'], text=True),
               'build_env': {k: os.environ.get(k, '') for k in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
                             'CARGO_PROFILE_RELEASE_DEBUG', 'CARGO_PROFILE_RELEASE_LTO']},
               'strip_version': subprocess.check_output([strip, '--version'], text=True).splitlines()[0],
               'model_sha256_gzip': sha256(Path('crates/zensally-tract/models/microsalnet_v3ds_w24.onnx.gz')),
               'measurements': rows}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
