# Preprocessing probe

This development-only workspace profiles the existing `preprocess_nchw` operation.
It does not change preprocessing, model weights, normalization, or detection thresholds.

```
just coz-build
just coz-run "$HOME/tmp/zensally-preprocess.jsonl" 1920 1080 320 240 rgba letterbox unit 1000
```

Use `run-heavy` and serialize with other heavy work. The profiler must be installed;
Coz progress means a complete preprocessing operation. The worker accepts source width
and height, target width and height, `rgb|rgba|bgra`, `stretch|letterbox`,
`unit|center|mean`, iterations per batch, and an optional new dump path.

Input is deterministic synthetic bytes (LCG seed `0x19b47e93`), not a face corpus.
Five operations warm up; `ZENSALLY_COZ_WARMUP_MS` optionally adds a fixed wall-time
warmup. Three `elapsed_ns` rows include preprocessing only, with reused output.
Validation outside timing SHA256-hashes every little-endian f32 bit of letterbox
metadata followed by the full planar tensor. An optional dump preserves those bytes.
The external `sha256sum` child is joined and excludes Coz injection.

These timings exclude inference and postprocessing. They cannot establish a whole
detector speedup or accuracy. The parent workspace's initial `--locked` clippy
command requires a lockfile update at revision `d257fe9d`; this isolated workspace
checks the unchanged core without resolving the backend graph. Full backend gates
remain necessary before accepting a production optimization.

`just coz-core-test` compiles the actual core crate root as a test target and
runs its existing assertions unchanged. It excludes optional serde/zenlayout
features and inference backends; this is a scoped gate, not workspace coverage.
