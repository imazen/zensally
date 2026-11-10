#!/usr/bin/env python3
"""Read complete zenbench packets; retain rejected attempts without filtering rounds."""
import argparse
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifacts", type=Path)
parser.add_argument("data", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()


def packet(log_name):
    path = args.artifacts / log_name
    log = path.read_text()
    match = re.search(r"(?:aggregate )?evidence → (.*)", log)
    pointer = json.loads((args.data / "runs" / Path(match[1]).name).read_text())
    compressed = (args.data / "objects" / f"{pointer['sha256']}.json.gz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == pointer["sha256"]
    raw = gzip.decompress(compressed)
    value = json.loads(raw)
    return value, {
        "log": log_name,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "packet_sha256": pointer["sha256"],
        "executable_sha256": value["execution"]["binary_sha256"],
        "valid": not (value["unreliable"] or value["incomplete"]),
        "gate_waits": value["gate_waits"],
        "wrapper": next(s for s in log.splitlines() if s.startswith("run-heavy: done")),
    }


result = {
    "schema_version": 1,
    "scaling_source": "6f0c862d5416ea21988603576b1889e4d09aa7cf",
    "refined_tiling_source": "b6e32879583e961a337faf2ca831744c1c992871",
    "batch_images": 32,
    "scaling": {},
    "all_scaling_attempts": {},
    "tiling": {},
    "profiles": {},
}
selected = {
    1: "scaling-1-diagnostic.log", 2: "scaling-2-quiet.log",
    4: "scaling-4.log", 8: "scaling-8.log",
}
for width, name in selected.items():
    value, record = packet(name)
    assert record["valid"] and len(value["replicates"]) == 5
    means = []
    for child in value["replicates"]:
        assert not (child["unreliable"] or child["incomplete"])
        assert child["cpu_placement"]["status"] == "enforced_uniform"
        group = child["comparisons"][0]
        assert group["completed_rounds"] == 10
        assert all(s["gate_clean"] for s in group["samples"])
        means.append(statistics.mean(s["compensated_ns"][0] / s["iterations"] for s in group["samples"]))
    mean = statistics.mean(means)
    record.update(mean_batch_ns=mean, process_mean_batch_ns=means, images_per_second=32e9 / mean)
    result["scaling"][str(width)] = record
for record in result["scaling"].values():
    record["descriptive_speedup"] = result["scaling"]["1"]["mean_batch_ns"] / record["mean_batch_ns"]
for path in sorted(args.artifacts.glob("scaling-*.log")):
    _, record = packet(path.name)
    result["all_scaling_attempts"][path.name] = record
for name in ["control-confirm.log", "tiles-confirm.log", "load-ahead-confirm.log",
             "tiles-confirm-traced.log", "load-ahead-confirm-traced.log"]:
    value, record = packet(name)
    log = (args.artifacts / name).read_text()
    record["comparison"] = next((s for s in log.splitlines() if s.startswith("process_comparison")), None)
    record["descriptive_means"] = [s.strip() for s in log.splitlines() if "sigmoid" in s and "█" in s]
    result["tiling"][name] = record
for width in [1, 8]:
    prefix = args.artifacts / "profiles" / f"threads-{width}"
    counters = prefix.with_suffix(".stat").read_text()
    usage = prefix.with_suffix(".time").read_text()
    result["profiles"][str(width)] = {
        "memory_bound_slots_percent": float(re.search(r"([\d.]+) %\s+tma_memory_bound", counters)[1]),
        "core_bound_slots_percent": float(re.search(r"([\d.]+) %\s+tma_core_bound", counters)[1]),
        "process_max_rss_kib": int(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", usage)[1]),
        "minor_page_faults": int(re.search(r"Minor \(reclaiming a frame\) page faults:\s*(\d+)", usage)[1]),
        "system_cpu_seconds": float(re.search(r"System time \(seconds\):\s*([\d.]+)", usage)[1]),
    }
with args.output.open("x") as output:
    json.dump(result, output, indent=2)
    output.write("\n")
