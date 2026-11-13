#!/usr/bin/env python3
"""Score a caller-supplied, human-labeled local manifest; calibrate separately."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import NormalDist

import evaluate


def validate(data):
    if not data.get("policy") or not data.get("annotation_provenance"):
        raise ValueError("declare policy and human annotation_provenance")
    hashes = set()
    families = {}
    counts = Counter()
    for row in data["items"]:
        if row["split"] not in ("calibration", "eval") or type(row["label"]) is not bool:
            raise ValueError("split must be calibration/eval and label must be boolean")
        for key in ("path", "sha256", "family", "content_type", "source_url", "license"):
            if not row.get(key):
                raise ValueError(f"missing item field: {key}")
        if row["sha256"] in hashes:
            raise ValueError("duplicate image hash")
        hashes.add(row["sha256"])
        if row["family"] in families and families[row["family"]] != row["split"]:
            raise ValueError("source family crosses calibration/eval split")
        families[row["family"]] = row["split"]
        counts[row["split"], row["label"]] += 1
    if any(counts[split, label] == 0 for split in ("calibration", "eval") for label in (True, False)):
        raise ValueError("both splits require positive and negative human labels")


def threshold(calibration, fpr):
    if not 0 <= fpr < 1:
        raise ValueError("FPR target must lie in [0,1)")
    negatives = sorted((r["score"] for r in calibration if not r["label"]), reverse=True)
    if not negatives:
        raise ValueError("no calibration negatives")
    allowed = math.floor(fpr * len(negatives))
    # Exclude the first negative beyond the allowed count, including all ties.
    return math.nextafter(negatives[allowed], math.inf)


def interval(k, n):
    if n == 0:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0, center - half), min(1, center + half)]


def metrics(rows, cut):
    tp = sum(r["label"] and r["score"] >= cut for r in rows)
    fp = sum(not r["label"] and r["score"] >= cut for r in rows)
    positives = sum(r["label"] for r in rows)
    negatives = len(rows) - positives
    return dict(tp=tp, fp=fp, fn=positives-tp, tn=negatives-fp,
                recall=tp/positives if positives else None,
                recall_wilson95=interval(tp, positives),
                false_positive_rate=fp/negatives if negatives else None,
                false_positive_rate_wilson95=interval(fp, negatives))


def run(root, manifest, output, fprs):
    import numpy as np
    from PIL import Image, ImageOps

    data = json.loads(manifest.read_text())
    validate(data)
    for item in data["items"]:
        if evaluate.digest(item["path"]) != item["sha256"]:
            raise ValueError(f"changed input: {item['path']}")
    output.mkdir(parents=True, exist_ok=False)
    evaluate.runtime(1)
    results = {}
    for key in evaluate.MODELS:
        directory = root / "models" / key
        cfg = json.loads((directory / "config.json").read_text())
        labels = cfg.get("label_names", cfg["pretrained_cfg"].get("label_names"))
        transform = evaluate.preprocessing(cfg)
        session = evaluate.session(directory / "native-fp32.onnx", 1)
        rows = []
        with (output / (key + "-scores.jsonl")).open("x") as f:
            for i, item in enumerate(data["items"]):
                with Image.open(item["path"]) as image:
                    pixels = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).numpy()
                logits = session.run(None, {"pixels": pixels})[0][0]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                pos_label = data.get("positive_label")
                neg_label = data.get("negative_label", "neutral")
                if pos_label and pos_label in labels:
                    score = float(probs[labels.index(pos_label)])
                elif neg_label in labels:
                    score = float(1 - probs[labels.index(neg_label)])
                else:
                    score = float(probs[0])
                row = dict(item, score=score, probabilities=dict(zip(labels, map(float, probs))))
                f.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if i % 32 == 0:
                    f.flush()
                    print(f"{key}: {i+1}/{len(data['items'])}", flush=True)
        calibration = [r for r in rows if r["split"] == "calibration"]
        evaluation = [r for r in rows if r["split"] == "eval"]
        points = []
        for target in fprs:
            cut = threshold(calibration, target)
            slices = {name: metrics([r for r in evaluation if r["content_type"] == name], cut)
                      for name in sorted({r["content_type"] for r in evaluation})}
            points.append(dict(calibration_fpr_target=target, threshold=cut,
                               calibration=metrics(calibration, cut), evaluation=metrics(evaluation, cut), slices=slices))
        results[key] = dict(operating_points=points, graph_sha256=evaluate.digest(directory/"native-fp32.onnx"))
        evaluate.save(output/(key+"-metrics.json"), results[key])
    evaluate.save(output/"summary.json", dict(policy=data["policy"], annotation_provenance=data["annotation_provenance"],
        manifest_sha256=evaluate.digest(manifest), environment=evaluate.environment(), models=results))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fpr", type=float, action="append")
    a = p.parse_args()
    run(a.root, a.manifest, a.output, a.fpr or [0.01, 0.05])
