# Local moderation: CPU runtime and latency comparison

Classifier accuracy and category recall remain **unmeasured**. Suitable labeled
evaluation data was not available locally; the checked public benchmark datasets
require access approval or credentials. These results select a speed candidate,
not a validated policy filter or a quality winner.

**SwiftFormer XS is the first CPU candidate to evaluate for quality.** All four
models load in tract and ONNX Runtime. Across eight existing repository photos,
all runtime probabilities agree with the native FP32 references within the
predeclared `0.0001` tolerance. This does not establish detection recall.

## Measured results

Host: i265, Intel Core Ultra 7 265K; one enforced performance core (CPU 0), FP32,
batch one. Four fresh process blocks per model/case, four warmups and sixteen
timed calls per process, cycling eight photos. A/A controls are retained separately.

| Model | ORT inference p50 / p95 ms | tract inference p50 / p95 ms | zensally-micro inference p50 / p95 ms | tract / zensally-micro peak MiB |
|---|---:|---:|---:|---:|
| SwiftFormer XS | 11.005 / 11.397 | 16.833 / 17.339 | **9.144 / 9.419** | 62.5 / 57.2 |
| SwiftFormer S | 16.769 / 17.334 | 23.366 / 24.072 | **14.729 / 15.183** | 81.8 / 88.1 |
| Marqo | 70.124 / 71.207 | 99.218 / 101.824 | — | 90.6 / — |
| Freepik | 1569.341 / 1587.642 | 1972.843 / 1991.652 | — | 786.3 / — |

The Rust path executes `zensally_micro::swiftformer::SwiftFormerModel` (zero-allocation
NHWC AVX2+FMA executor via `archmage` / `magetypes`) for SwiftFormer XS and S, and
`tract 0.23.10` for the baseline and non-SwiftFormer models (`marqo`, `freepik`). The ORT
path uses ONNX Runtime 1.31.1 through Python. These are API-call timings inside persistent
model sessions; process startup, graph optimization and input file reads are excluded.

| Model | ORT preprocessing + inference p50 / p95 ms | ORT JPEG decode + preprocessing + inference p50 / p95 ms |
|---|---:|---:|
| SwiftFormer XS | 17.680 / 18.758 | 29.923 / 45.378 |
| SwiftFormer S | 23.308 / 24.697 | 39.393 / 51.726 |
| Marqo | 78.377 / 80.005 | 94.433 / 106.892 |
| Freepik | 1576.618 / 1589.504 | 1591.763 / 1604.705 |

Decoding starts from JPEG bytes already in memory. The eight source photos have
different dimensions; these totals are specific to that manifest. Rust decoding
and preprocessing were not timed. GPU, BF16 and INT8 execution were not measured.

Each table entry summarizes 64 calls in four process clusters. P50/p95 are
descriptive call percentiles, not confidence intervals. Zenbench's Iterate mode
completed all 12 ORT blocks and all four tract blocks with correctness checks and
enforced uniform placement. No Confirm-mode speedup verdict or production tail
latency guarantee is claimed. The machine was shared, with observations retained.

Peak MiB comes from `/usr/bin/time -v` for the complete worker, including model
loading and graph optimization. The Python workers also load PyTorch for
preprocessing; their peaks are not ONNX Runtime alone. Resource-wrapper receipts:

```text
ORT:            run-heavy: done rc=0 545s | peak-RSS 1.69GiB | min-avail 20730MiB | peak-load 1.55
tract:          run-heavy: done rc=0 180s | peak-RSS 0.77GiB | min-avail 21437MiB | peak-load 1.29
zensally-micro: run-heavy: done rc=0 26s  | peak-RSS 0.23GiB | min-avail 21872MiB | peak-load 0.41
```

## SwiftFormer XS/S CPU profiling and `zensally-micro` optimization

Before optimizing SwiftFormer XS (`3.26M` params, `603.57M` MACs) and SwiftFormer S
(`5.98M` params, `985.79M` MACs), both `tract 0.23.10` and `ONNX Runtime 1.31.1` were
profiled on CPU 0 (`i265`) using `valgrind --tool=callgrind`, Linux `perf stat`, and ORT's
built-in node execution profiler (retained under
`/mnt/v/output/zensally/moderation-swiftformer-opt-20261113/`):

1. **Pointwise `1x1` convolutions (`ConvEncoder.pwconv1/2`, `Mlp.fc1/2`, `LocalRepresentation.pwconv1/2`, `to_query/to_key/proj/final_proj`)** represent **85.06% (XS)** / **89.00% (S)** of total MACs (`513.41M` / `877.35M`) and **62.01% (XS)** / **64.88% (S)** of `tract` dynamic instructions (`fma_mmm_f32_16x6` + `f32_4x4`).
2. **Dense `3x3` stride-2 convolutions (`stem.0`, `stem.3`, `stages.{1,2,3}.downsample`)** represent **13.51% (XS)** / **9.93% (S)** of MACs (`81.54M` / `97.88M`) and `8.88%` (XS) / `7.72%` (S) of ORT runtime.
3. **Unfused `GELU` (`Erf`) + L1d-exceeding `4x` intermediate tensors**: in Stage 0 (`56x56`), the `4x` expanded tensor `[1, 192, 56, 56]` is **2.41 MiB** (exceeding both L1d and L2 on a single core). Both `tract` and ORT materialize the full tensor before running a separate `Gelu` (`Erf`) pass (`17.20%` / `1.974 ms` of ORT runtime on XS; `14.21%` / `2.432 ms` on S) and re-reading it in `pwconv2` (`Add` takes another `10.17%` of `tract` instructions).
4. **Layout transposes & unfused norms around `EfficientAdditiveAttention`**: ORT spends **16.30% (`1.872 ms/run` XS)** and **12.65% (`2.165 ms/run` S)** in `EfficientAdditiveAttention` overhead (`ReorderOutput`, `Transpose`, `Gemm`, `ReduceL2`, `Clip`, `Expand`, `Div`, `Mul`, `ReduceSum`, `BatchNormalization`, `ReorderInput`) converting between `Nchwc` conv blocks and `[B, N, D]` token blocks.

### Implemented `zensally_micro::swiftformer::SwiftFormerModel` architecture

- **Unified NHWC (`[H*W, C_pad]`) token layout**: every `ConvEncoder`, `Downsample`, `LocalRepresentation`, `EfficientAdditiveAttention`, and `Mlp` block operates on `[N, C_pad]` rows with zero layout transposes.
- **Load-time FP64 parameter folding**: all `BatchNorm2d` layers (including `Block.linear.norm1` before `fc1`), all `LayerScale2d` (`gamma`) multipliers, `w_g * scale_factor`, and final `BatchNorm1d + ReduceMean(1/49) + 0.5*(head + head_dist)` are folded in `f64` at load time before packing to `f32`.
- **Zero-padded multiple-of-8 channel storage (`220 -> 224` in XS Stage 3)**: output weights and biases for channels `220..224` are zero-padded so every channel dimension is a multiple of 8 (`f32x8`). Because `GELU(0.0) == 0.0` and `0.0^2 == 0.0`, padded channels remain identically zero through `dwconv`, `GELU`, and `ReduceL2` while keeping the exact `scale_factor = 220^-0.5`.
- **L1d token tiling (`M_TILE = 24`) + `6x16` / `4x16` FMA microkernel + 15-register 2x-unrolled `gelu_exact_in_place`**: tiles `pwconv1 -> GeluExact -> pwconv2 (+ Residual)` over 24 tokens at a time (`18–86 KiB`, staying in L1d/L1.5 cache) and fuses residual accumulation into the `pwconv2` epilogue.

### Direct interleaved A/B `zenbench` and hardware PMU comparison (CPU 0)

| Metric | SwiftFormer XS (`tract` -> `zensally-micro`) | SwiftFormer S (`tract` -> `zensally-micro`) |
|---|---|---|
| Pooled `zenbench` p50 / p95 (`pair-swiftformer-opt-v1`) | `16.833 / 17.339 ms` -> **`9.144 / 9.419 ms`** (**1.84x faster vs tract, 1.20x faster vs ORT**) | `23.366 / 24.072 ms` -> **`14.729 / 15.183 ms`** (**1.59x faster vs tract, 1.14x faster vs ORT**) |
| Direct interleaved A/B `zenbench` (`pair-ab-swift-{xs,s}-v1`) | `16.732 / 17.212 ms` -> **`9.155 / 9.428 ms`** (`latency_change_pct = -45.22%`, `paired_ratio` median `0.5467`) | `23.786 / 24.595 ms` -> **`15.003 / 15.713 ms`** (`latency_change_pct = -36.50%`, `paired_ratio` median `0.6344`) |
| Callgrind dynamic instructions (`Ir`, 20 runs) | `6,636,082,753` -> **`4,836,123,818`** (`-27.1%`, `96.49%` in `__arcane_forward_impl_v3`) | `9,350,489,447` -> **`7,578,637,060`** (`-18.9%`, `96.50%` in `__arcane_forward_impl_v3`) |
| `perf stat` CPU cycles (20 runs, CPU 0) | `2,249,289,389` -> **`1,101,718,362`** (**`-51.0%`**) | `3,187,729,732` -> **`1,798,084,033`** (**`-43.6%`**) |
| `perf stat` IPC & `tma_retiring` | IPC `2.9` -> **`4.4`**, retiring `35.9%` -> **`51.8%`** | IPC `2.9` -> **`4.2`**, retiring `35.7%` -> **`49.6%`** |
| Max abs probability diff vs native FP32 (8 witnesses) | **`0.0000023909`** (`<= 0.0001` PASS) | **`0.0000003348`** (`<= 0.0001` PASS) |

## Correctness and preprocessing findings

The largest probability error across all three backends (`onnxruntime`, `tract`, and
`zensally-micro`) and all eight native references was `0.000002390936985396941`, well below
`0.0001`. Model identities and publisher ONNX hashes are in the
[assessment](../docs/CONTENT_MODERATION.md).

SwiftFormer's documented bilinear squash path differs from its native
bicubic/center-crop path: the largest class-probability differences on the eight
witnesses were `0.09387022256851196` (XS) and `0.1452409029006958` (S). Matching
the native crop/resample pixels restores publisher-graph parity. The measured
comparison therefore uses each model's native preprocessing contract.

Freepik's fused CPU attention could not be exported by the torch 2.15 legacy
exporter (`is_causal` was traced as a Tensor). Timm's explicit attention was used
for both its FP32 native reference and export, with unchanged model weights.
Its original safetensors are BF16; these timings cover FP32 execution only.

The Rust example passes Clippy with warnings denied and contains no unsafe code.
Three calibration tests cover tied scores, confusion counts and split leakage.
They validate evaluation arithmetic, not classifier quality. No existing test
expectation was relaxed.

Repository CI was already failing on the pre-evaluation commit `2c39fb8f`:
Cargo 1.91 rejects the micro crate's workspace `default-features` override, and
the i686 tier-parity test requires at least two permutations while that target
runs one (the following allocation test then observes a poisoned mutex). These
existing gates remain unresolved; the local checks above do not imply green CI.
[Pre-evaluation CI run](https://github.com/imazen/zensally/actions/runs/43333029596).

## Accuracy gate and recommendation

Keep XS and S for the labeled comparison. XS is fastest in both measured runtimes;
S's extra CPU cost needs a demonstrated reduction in misses or false positives.
Marqo and Freepik remain quality comparators, but their CPU costs are measured
above and no quality advantage was established here.

The [accuracy runner](../tools/moderation-eval/README.md) is ready for a licensed,
human-labeled manifest with calibration/evaluation splits and source-family
separation. It freezes thresholds on calibration negatives and reports held-out
confusion counts and Wilson intervals. No threshold was selected from these
eight unlabeled photos, and no deployment threshold is recommended.

The checked gated benchmark dataset on Hugging Face lists 2,037 test items;
its pinned README request returned HTTP 401 without a configured credential.
No labeled evaluation corpus was found in the checked local data indices.
Single-category evaluation data alone would still leave multi-category policy recall
unproven. Known-content perceptual hash matching and its reference index are outside
these measurements.

## Reproduction and retained evidence

Source: Python worker at `573fd6d5`, Rust example at `e3b10747`; model, graph,
worker and binary hashes are retained with the attempts. Rust 1.99.1, release
profile, default target; no target-cpu override was configured. Timm 1.0.31,
torch 2.15.0+cu132, ONNX Runtime 1.31.1; actual imported NumPy was 2.5.2 through an
inherited PYTHONPATH, while venv metadata listed 2.5.4. Actual module locations,
package metadata and environment freeze are retained rather than conflated.

Commands are the `moderation-*` recipes in the [justfile](../justfile).
`plan-ort-v3.json` and `plan-tract-v1.json` freeze the measured workloads.
`report.py` refuses incomplete attempts, excludes A/A from baseline pooling,
checks probabilities against all native references, and exports the
[scalar table](local_moderation_2026-11-13.csv).

Artifact directory, checked on i265:
`/mnt/v/output/zensally/moderation-20261113/`.
It contains pinned weights/configs, FP32 exports, input hashes, probe results,
retained release binary, both complete zenbench packets, raw per-process data,
memory receipts, logs and `summary.json`. Initial failed setup/export attempts
and the successful smoke check remain separate and are excluded from the tables.

Verified mirror:
`tower:/mnt/disk1/coefficient/output/zensally/moderation-20261113-evidence.tar.zst`.
Archive: 465,387,589 bytes; SHA-256
`d005cae47f3c2f7e70e69dfb51d71ca353f60e05e692aef4bff92d30b0c7e0b6`.
The remote archive hash and size match the local archive. Three seeded small-file
readbacks from that byte-identical archive match their source hashes; receipts
are retained locally in `BACKUP_RECEIPT.json`.
