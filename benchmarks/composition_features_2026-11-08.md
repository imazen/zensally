# Local composition-feature suitability — 2026-11-08

No tested feature variant improves the frozen crop initializer on TRAIN-development
loss. These are negative suitability results, not a promoted model. Canonical
VALIDATE/TEST remain unopened during selection; no paid model API was called.

## Matched frozen-encoder heads

The [registered experiment](../docs/experiments/COMPOSITION_FEATURE_ABLATION.md)
uses 64 fitting and 16 development families with existing noisy rendered-image
judgments. All five heads have 76,514 parameters and initially reproduce the
frozen scorer exactly. Seeds 31/137/911 each run 60 epochs.

| Inputs | Mean selected development loss |
|---|---:|
| Frozen FP32 initializer | 0.9299975634 |
| Continued head fitting | 0.9382486343 |
| Pose | 0.9383482337 |
| Pose + subject gaze | 0.9383175174 |
| Pose + gaze + viewer scanpaths | 0.9377966126 |
| Source-permuted features | 0.9385672808 |

All 15 runs complete. The full feature arm only narrowly improves the
continued-fitting control; none improves the frozen initializer's loss.
Across the nine essential ratios and both crop modes, pooled across three seeds,
the continued baseline offers 401 crops with 257 cached acceptable judgments;
the full arm offers 403 with 258. These are 864 source-seed requests on selection
data, not independent trials or human composition ratings. Saved bootstrap
summaries resample the 16 source families with seeds averaged within family.

Pose is present in only 4/64 fitting sources and 4/16 development sources.
Subject-gaze heads are available in 6/64 and 4/16. In contrast, the separate 56
synthetic photos have pose in 43 and gaze heads in 46. The first gate keeps these
56 annotation-only. Missing detection never certifies absence or face safety.

## End-to-end geometry distillation

A separately registered successor admits synthetic photos only for geometry
distillation. Shared fictional brands and scenario variants stay in one family:
48 fitting sources in 27 families; eight development sources in eight families.
The auxiliary development set has no branded commerce, an important limitation.
Old composition labels and all canonical source roles remain unchanged.

Nine matched fits complete: baseline, geometry and source-permuted geometry,
three seeds, 40 epochs, original encoder trainable with BatchNorm running
statistics frozen. **All nine select epoch zero**, development loss
0.9300000668 in this batch/layout condition. This number is not substituted for
the separately measured frozen-descriptor value above.

Final-epoch normalized teacher-feature errors, means across three seeds:

| Arm | Pose MAE | Subject-gaze MAE | Viewer-path MAE | Crop dev loss |
|---|---:|---:|---:|---:|
| Continued baseline | 0.73105669 | 0.70809803 | 0.79944817 | 0.98701445 |
| Geometry | 0.40747119 | 0.41513016 | 0.40945758 | 0.99895038 |
| Permuted geometry | 0.41642319 | 0.43471192 | 0.41315836 | 0.99635154 |

Lower auxiliary error alone does not establish useful image understanding:
the permuted control is close, and many target features depend strongly on crop
geometry. No end-to-end checkpoint is exported or promoted.

## Local teachers and critic

All 136 fit/development/annotation sources have compact MediaPipe geometry,
Gaze-LLE subject-gaze targets, and DeepGaze III viewer paths. Four sampled paths
contain six free fixations each, after an imposed central start. They model
viewer attention under those assumptions, not a uniquely correct artistic crop.
Only coordinates, confidence and candidate scalars are retained; no new
diagnostic maps are saved. Eight photographic witnesses pass browser decoding
and all three overlay controls without JavaScript errors.

PhotoFramer's pinned 7B composition model is tested locally in NF4 with BF16
compute. All 16 outputs parse: four synthetic originals and six fixed paired
TRAIN-development crop comparisons. It agrees with cached accept/reject ordering
on three pairs, ties on two and reverses one. This small weak-label comparison
is insufficient for trustworthy composition supervision. Ratings have not been
admitted as training labels. Quantization/reference parity is not established. A second run keeps the vision
encoder in BF16 and obtains the same aggregate three agreements/two ties/one
reversal, with different individual scores. It completes 16/16 outputs in 31 s,
run-heavy peak RSS 15.43 GiB, minimum available 24,430 MiB and peak load 12.75.
A failed skip-module attempt is retained separately.

The original checkpoint's null `use_cache` fails current Transformers validation.
The runner preserves original bytes and supplies an explicit config plus a
derived processor config changing that field to true. Failed attempts remain
retained. No model weights or teacher dependencies enter the shipping crate.

## Resources and verification

Measurements use the RTX 4080 on i265, FP32/TF32-off for feature fitting, two CPU
threads and explicit device-cache/GPU-memory bounds. Complete wrapper logs:

| Work | Wrapper elapsed | run-heavy peak RSS | CUDA allocated peak |
|---|---:|---:|---:|
| Gaze, 136 sources | 5 s | 1.52 GiB | 441,747,456 B |
| Scanpaths, 136 sources | 92 s | 1.78 GiB | 479,001,088 B |
| Frozen descriptors | 3 s | 1.34 GiB | not reported |
| 15 matched head fits | 7 s | 1.36 GiB | 63,095,296 B |
| 9 joint fits | 30 s | 1.91 GiB | 601,936,896 B |
| NF4 critic, 15 uncached images | 31 s | 9.90 GiB | 6,136,389,632 B |

These are complete workload durations, not isolated inference benchmarks.
Corresponding run-heavy minimum available memory: 21,871 / 21,669 / 23,900 /
23,063 / 24,218 / 24,620 MiB. Peak loads: 14.52 / 14.52 / 12.49 / 12.28 /
12.62 / 13.23. The critic's first image is a separate 13-second load/probe with
8.16 GiB peak RSS. Its 16.6-GB original weights are offline training candidates,
not a proposed runtime dependency.

All 91 retained frozen-head checkpoints now have immutable replayed predictions;
all 15 selected results reproduce exactly. An early writer retained every
checkpoint but replaced convenience prediction JSON; replay repairs epoch
evidence without refitting or changing selection. New runs retain every epoch
prediction. A new guard initially expected `None` for a disconnected concatenated
branch; PyTorch materializes zero gradients. The corrected guard checks zeros
and preserves all earlier contracts. Five new composition guards pass.

The [frozen evidence pointer](composition_features_2026-11-08.pointer.md) records
2,297 independently hash-verified NAS members. The next quality gate needs better photographic crop
preferences and independent face coverage, not more optimization of this tiny
weak-label bank.

http://dev.lan:3300/zensally/composition-features/2026-11-08/witness/index.html
