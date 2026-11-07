#!/usr/bin/env python3
"""Report resolved Cargo graph counts and embedded model bytes; no builds."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def cargo_dependencies(package, features=()):
    command = [
        "cargo", "tree", "--locked", "-p", package,
        "-e", "normal,build", "--prefix", "none", "--format", "{p}",
    ]
    if features:
        command += ["--features", ",".join(features)]
    lines = subprocess.check_output(command, text=True).splitlines()
    # Cargo repeats a node with (*) when its subtree was already displayed.
    packages = {line.removesuffix(" (*)") for line in lines}
    packages.discard(lines[0])
    return sorted(packages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    graphs = {}
    for label, package, features in [
        ("core", "zensally", ()),
        ("core + serde", "zensally", ("zensally/serde",)),
        ("core + zenlayout", "zensally", ("zensally/zenlayout",)),
        ("tract", "zensally-tract", ()),
        ("fixed w24", "zensally-micro", ()),
        ("depth cue only", "zensally-depth", ()),
        ("embedded depth + JPEG + HEIC", "zensally-depth", ("zensally-depth/jpeg", "zensally-depth/heic")),
        ("plugin host", "zensally-zentract", ()),
    ]:
        graphs[label] = cargo_dependencies(package, features)
    models = []
    for path in sorted((root / "crates/zensally-tract/models").glob("*.gz")):
        compressed = path.read_bytes()
        onnx = gzip.decompress(compressed)
        models.append({
            "model": path.name,
            "gzip_bytes": len(compressed),
            "onnx_bytes": len(onnx),
            "sha256_gzip": hashlib.sha256(compressed).hexdigest(),
        })
    if args.json:
        print(json.dumps({"dependencies": graphs, "models": models}, indent=2))
    else:
        print("Unique resolved normal/build dependencies (root excluded; host target):")
        for name, packages in graphs.items():
            print(f"  {name}: {len(packages)}")
        print("Embedded models (gzip bytes / ONNX bytes):")
        for model in models:
            print(f"  {model['model']}: {model['gzip_bytes']} / {model['onnx_bytes']}")


if __name__ == "__main__":
    main()
