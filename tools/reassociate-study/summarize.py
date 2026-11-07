#!/usr/bin/env python3
"""Summarize retained logs and complete frozen geometry reports, without fitting."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifacts", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
result = {
    "schema_version": 1,
    "kernel_source": "ed0ce59878eacfcaeac538d6051725fe6436353c",
    "baseline_source": "4dd26ecc24ab68ecba133c7688d20ee6fdbec9fa",
    "confirmation": {},
    "geometry": {},
}
for name in ["control", "split", "algebraic", "depthwise", "combined"]:
    phases = ["confirm"] + (["long-confirm"] if name != "split" else [])
    for phase in phases:
        path = args.artifacts / f"{name}-{phase}.log"
        log = path.read_text()
        lines = log.splitlines()
        result["confirmation"][f"{name}-{phase}"] = {
            "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "invalid": "INVALID confirmation" in log,
            "comparison": next((s for s in lines if s.startswith("process_comparison")), None),
            "descriptive_means": [s.strip() for s in lines if "sigmoid" in s and "█" in s],
            "wrapper": next(s for s in lines if s.startswith("run-heavy: done")),
            "aggregate_reference": re.search(r"aggregate evidence → (.*)", log)[1],
        }
    if name == "control":
        continue
    path = args.artifacts / f"{name}-rectangles.tsv"
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    if len(rows) != 1984:
        raise ValueError(f"Incomplete frozen geometry report: {path}: {len(rows)}")
    values = [float(row[k]) for row in rows for k in ["max_map_abs_delta", "roi_edge_map_px", "crop_edge_map_px"]]
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"Nonfinite geometry/numerical result: {path}")
    result["geometry"][name] = {
        "comparisons_including_ordered_control": len(rows),
        "experimental_comparisons": sum(row["candidate"] == "fast-sigmoid" for row in rows),
        "max_map_abs_delta": max(float(row["max_map_abs_delta"]) for row in rows),
        "max_roi_map_pixels": max(float(row["roi_edge_map_px"]) for row in rows),
        "max_crop_map_pixels": max(float(row["crop_edge_map_px"]) for row in rows),
        "presence_mismatches": sum(row["presence_mismatch"] != "false" for row in rows),
        "outside_budget": sum(row["within_budget"] != "true" for row in rows),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
with args.output.open("x") as output:
    json.dump(result, output, indent=2)
    output.write("\n")
