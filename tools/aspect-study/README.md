# Aspect and crop refinement study

The private `aspect_crops` example compares six fixed image-space strategies
without changing the deployed model API. Protocol and limits:
[ASPECT_CROPS_PROTOCOL.md](../../docs/experiments/ASPECT_CROPS_PROTOCOL.md).

Run `just aspect-check`, freeze existing DUTS-TE pairs with `freeze.py`, create
four source-disjoint shards with `just aspect-shards`, and run
`just aspect-run EXECUTABLE DATASET SHARDS NEW_OUTPUT` through run-heavy with
6 GiB and four jobs. The source data and output directories must be explicit.
Missing files, changed hashes, malformed masks or invalid crops are hard errors.
Each child keeps a complete log and completion marker; raw outputs stay outside
git. `MAX_SOURCES` is an explicit smoke-run limit, never an implicit skip.

Before reading quality results, freeze this policy-selection rule: choose the
method with lowest **calibration mean Minimal crop retention regret** in each
source aspect bin [1,1.5), [1.5,2), [2,4), [4,infinity). Break exact numerical
ties at 1e-6 by fewer model passes, then method name. Natural and synthetic
calibrations remain separate. Require at least 30 distinct calibration sources
per bin, else use stretch. Evaluate this choice once on evaluation sources.
No per-image mask or label enters the prediction path. Report all fixed arms,
not only the selected policy; do not treat stress variants as independent
sources. Bootstrap paired differences at source level with seed 20261107 and
2,000 replicates. Report natural and stress separately, including aspect bins,
Minimal/Maximal crop retention, regret and crop-area fraction. This mask
protocol does not measure human aesthetic preference.

The gallery preserves the source/mask, full maps and actual extracted crops.
Read generated PNGs with an independent decoder and verify dimensions and
mask-grid interpretation. Python Pillow is an oracle only; it does not write
study images. Tables over 50 MB are converted to Parquet+zstd; original shard
outputs and failed attempts are preserved. Cite dataset and paper URLs in
reports. No images or generated binary tables belong in git.
