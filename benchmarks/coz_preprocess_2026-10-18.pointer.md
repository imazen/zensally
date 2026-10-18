# Preprocessing Coz probe, 2026-10-18

Production core: `d257fe9d` (unchanged). Worker source introduced in `5da648434a56481d25653fa229cd2f3c3ed6869e`.
Host: r7900x, 24 logical CPUs. No `target-cpu=native`.
Artifacts checked on dev: `/mnt/v/output/zensally/coz-2026-10-18-r7900x/`.
The original scratch copy is on r7900x, which has no `/mnt/v` mount.
No R2 or tower mirror has been created; retain both copies.

Command, inside `flock ~/tmp/zen-coz-heavy.lock` and `run-heavy --mem 16G --jobs 8`:

```
coz run --source-scope '%/zensally/src/%' --output preprocess-probe.jsonl --- \
  tools/coz-profile/target/release/zensally-coz-profile \
  1920 1080 320 240 rgba letterbox unit 1000
```

Three source-selected experiments have positive selected samples, with 2,950
completed operations across their progress intervals. This verifies attribution
and completed-operation counters. It does not establish a causal response curve,
a performance improvement, inference cost, or whole-detector coverage. Timings
under injected virtual speedups are not baseline timings.

Probe output SHA256 (all tensor and metadata bits):
`94be27ff5f64b0bfc2dc7d55402c59dd7f520131f79ebcc5eb0b9424c872f81b`.
Two independent `65 33 32 24 bgra letterbox mean 1` dumps compare byte-for-byte:
`4f1ddd63aabd7445444eef5172e326c2f50f943c2a0e3e19c511d05fc03ed047`.

Checksums:

- Probe JSONL: `adddbb7872294d2c1f1a083f3d69221500710a30015d2fa31dc8c304df886dfe`.
- Worker executable: `89c5d26572e66c33cd6ee3b8ed22671219e65b6915365255263d6c32ee6b7624`.
- Production preprocess source: `8a29fea084359a3b5db7c81f4fa0a40bf4a0d0dc00b0216dedd64d3572a0fff8`.

Resource records (full logs in the artifact directory):

```
probe: run-heavy: done rc=0 1s | peak-RSS 0.02GiB | min-avail 22945MiB | peak-load 1.47
build: run-heavy: done rc=0 2s | peak-RSS 0.21GiB | min-avail 22842MiB | peak-load 1.60
unit:  run-heavy: done rc=0 0s | peak-RSS 0.20GiB | min-avail 22983MiB | peak-load 1.47
```

The isolated actual-source test target passes 19 unchanged core tests. It excludes
optional serde/zenlayout and inference backends. Full workspace `--locked` clippy
requires a lockfile update at the inspected revision; its failure log is retained.
Isolated worker/actual-source-test clippy and formatting pass. No runtime source,
weights, normalization, detector threshold, public API, or test expectation changed.
