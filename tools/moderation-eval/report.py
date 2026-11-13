#!/usr/bin/env python3
"""Summarize complete zenbench attempts; retain process boundaries and parity."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import statistics

import evaluate


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values)-1)*quantile
    lo = math.floor(position)
    return values[lo]+(values[math.ceil(position)]-values[lo])*(position-lo)


def summarize(root, attempts, output):
    groups = defaultdict(list)
    peaks = defaultdict(list)
    packet_records = []
    parity = defaultdict(float)
    for name in attempts:
        packet_path = root/name/"packet.json.gz"
        with gzip.open(packet_path, "rt") as f:
            packet = json.load(f)
        if not packet["complete"]:
            raise ValueError(f"incomplete attempt: {packet_path}")
        packet_records.append(dict(path=str(packet_path), sha256=evaluate.digest(packet_path),
                                   placement=packet["placement"], summaries=packet["summaries"]))
        arms = [packet["plan"]["baseline"], *packet["plan"]["candidates"]]
        for invocation in packet["invocations"]:
            if invocation["status"] != "completed":
                raise ValueError(f"failed invocation: {invocation}")
            if invocation["arm"] >= len(arms):
                continue  # A/A stays in the packet and is not pooled with baseline.
            arm = arms[invocation["arm"]]["id"]
            directory = Path(invocation["output"])
            worker = json.loads((directory/"worker.json").read_text())
            backend = worker.get("backend", "onnxruntime")
            model = worker.get("model", arm)
            group = (backend, worker["stage"], model)
            groups[group].append(worker["samples_ms"])
            text = (directory/"memory.txt").read_text()
            match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
            if not match:
                raise ValueError(f"missing OS memory measurement: {directory}")
            peaks[group].append(int(match[1]))
            reference_path = root/"probe-freepik-v2.json" if model == "freepik" else root/"probes-v1"/(model+".json")
            reference = json.loads(reference_path.read_text())["native_scores"]
            for i, logits in enumerate(worker["logits"]):
                values = logits[0]
                values = [math.exp(x-max(values)) for x in values]
                total = sum(values)
                difference = max(abs(x/total-y) for x,y in zip(values,reference[i % len(reference)],strict=True))
                parity[backend, model] = max(parity[backend, model], difference)
    rows = []
    for key, processes in sorted(groups.items()):
        if len(processes) != 4 or any(len(values) != 16 for values in processes):
            raise ValueError(f"wrong process/sample count: {key}")
        values = [v for process in processes for v in process]
        rows.append(dict(backend=key[0], stage=key[1], model=key[2], processes=4, calls=64,
                         per_call_p50_ms=statistics.median(values), per_call_p95_ms=percentile(values,.95),
                         process_mean_ms=[statistics.mean(v) for v in processes],
                         process_max_rss_kib=peaks[key], max_rss_mib=max(peaks[key])/1024))
    if any(error > .0001 for error in parity.values()):
        raise ValueError(f"runtime probability parity failed: {dict(parity)}")
    evaluate.save(output,dict(rows=rows, packets=packet_records,
        runtime_probability_max_abs_error={f"{backend}/{model}":v for (backend,model),v in parity.items()},
        accuracy_measured=False,category_recall_measured=False,
        statistics="Pooled per-call p50/p95 are descriptive; four process clusters retained. A/A excluded from baseline pooling. Exploratory, not a confirmed speedup."))
    for row in rows:
        print(f"{row['backend']:12} {row['stage']:28} {row['model']:8} p50={row['per_call_p50_ms']:.3f} ms p95={row['per_call_p95_ms']:.3f} ms peak={row['max_rss_mib']:.1f} MiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--attempt",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    summarize(a.root,a.attempt,a.output)
