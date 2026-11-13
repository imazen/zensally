# Local moderation comparison

Development-only tools. Four pinned classifiers; local CPU inference after model
download. No public API, shipping dependency or detection policy is introduced.

`evaluate.py fetch --root ROOT` retrieves configs, safetensors and any publisher
ONNX artifacts. Existing files are verified, partial downloads are preserved,
and LFS hashes are checked. `probe` loads weights strictly and compares a fresh
FP32 ONNX export with native inference on a hashed image manifest.

Manifest shape: `{"items": [{"path": "/absolute/image", "sha256": "..."}]}`.
These images can exercise correctness and latency without moderation labels.
They cannot establish detection quality. Accuracy evaluation needs a separate
licensed manifest with human labels and declared calibration/evaluation splits.

Use `worker` only under `zenbench pair` and the resource wrapper. Session creation,
disk I/O, warmup and validation are outside timing. Three stages cover inference,
decoded-RGB preprocessing plus inference, and in-memory JPEG decoding plus
preprocessing plus inference. Every process retains per-call timings and outputs.
`check` verifies matched workloads and exact same-model A/A results; different
models need not agree. The Rust example has a preprocessed inference timing mode.

`plan` freezes four blocks, four warmups and sixteen timed calls per process.
`--binary /absolute/moderation_probe` selects the Rust worker. `run` uses a
transient Linux system scope on CPU 0 with an 8 GiB cap and shared zenbench
admission; first verify CPU 0 is a suitable core on the deployment machine.
Use `report.py` to combine complete attempts, check runtime probability parity
against all eight native witnesses, and retain descriptive p50/p95 plus OS peak
RSS. Failed attempts are preserved and cannot contribute to the summary.

`accuracy.py` accepts a separately supplied human-labeled manifest:

```json
{
  "policy": "Describe the target classification policy and positive class definition",
  "annotation_provenance": "Human annotation process and version",
  "items": [{
    "path": "/absolute/image.jpg", "sha256": "image sha256",
    "family": "original-source-id", "content_type": "photo",
    "source_url": "https://example.org/original", "license": "license identifier",
    "split": "calibration", "label": true
  }]
}
```

Both splits (`calibration`, `eval`) require positive and negative labels. Duplicate
hashes and source families crossing splits are errors. Thresholds use calibration
negatives only, at caller-selected false-positive targets (defaults 1% and 5%).
Held-out outputs include confusion counts, Wilson intervals and content slices.
Each model's positive score is computed from its configured positive/negative label
mapping (`positive_label` or `1 - P(negative_label)`). The selected policy must
account for those label definitions. Run the calibration tests with
`just moderation-check PYTHON`.

See [the assessment](../../docs/CONTENT_MODERATION.md) for coverage limitations.
