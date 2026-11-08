# Primary-extent suitability — 2026-11-08

No small-model variant earns promotion. A larger frozen encoder shows a small,
uncertain weak-target signal; its first feature-hint distillation does not help
the small encoder. These are object-location diagnostics, not artistic crop
ratings. See the [registered protocol](../docs/experiments/SEMANTIC_CROP_SUCCESSOR.md).

Pinned Grounding DINO tiny revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71` has 172,250,626 parameters and a
689,359,096-byte safetensors file. The current Transformers reference runs
FP32/TF32-off without custom deformable kernels. Owned Lanczos RGB preserves
aspect at long axis 1024, with processor resizing disabled; upstream pixel
parity is not claimed. One source-specific primary-product/story query is
frozen for every synthetic image before full inference. At thresholds 0.3/0.25,
all 56 sources return at least one visible detection. That is response coverage,
not completeness, recall or semantic correctness. Raw/clipped boxes, scores,
labels, pixels and hashes are retained; diagnostic maps are not saved.

Targets are unions of the returned visible primary boxes. The student receives
no query; ambiguous intent and partial/false-positive detections limit this
supervision. These labels are admitted only for the registered weak extent
task, not composition acceptability or face certification. Source roles remain
48 fit images in 27 brand/scenario families and eight development images in
eight families. No branded commerce is in development. Canonical VALIDATE/TEST
remain unopened. No paid API is used.

## Matched models

The frozen small encoder provides 512 visual values from 4×4 pools of two
16-channel projections; four geometry values complete the 516→64→6 head
(33,478 parameters). Two three-way partitions produce finite visible nonempty
boxes, decoded in FP32. Fit-only feature normalization stays fixed.
Aspect-only heads zero visual inputs; permutation controls shuffle visual
inputs within role/aspect bins while keeping geometry/targets attached to their
source. Three seeds run 200 epochs. Epoch selection includes zero and uses
only the registered weak development objective. All runs retain checkpoints
and source-level predictions; the eight development families are selection
data, never independent confirmation.

| Experiment | Aspect-only IoU | Image IoU | Permuted IoU |
|---|---:|---:|---:|
| Frozen small features, edge MSE | 0.36907730 | 0.31281736 | 0.34967692 |
| Frozen small features, L1/GIoU | 0.41041508 | 0.33787065 | 0.40284353 |
| End-to-end small encoder, L1/GIoU | — | 0.37497610 | 0.44107033 |
| Frozen FocalNet features, L1/GIoU | 0.41041508 | 0.43076265 | 0.37538898 |

Numbers are mean selected development IoU across seeds 31/137/911. The fit-mean
box reaches 0.37582982 without image information. End-to-end fitting uses
198,762 parameters, frozen BatchNorm running statistics, AdamW encoder/head lr
0.0001/0.001, batch 16, weight decay 0.01, clipping 5, FP32/TF32-off. The objective
is 5× mean edge L1 + 2× generalized-IoU loss. Frozen-head runs use lr 0.001.

FocalNet preserves the same 512-scalar budget using its rank16 pool4×4,
fused48 pool2×2, 48 global means and 16 zeros. The aspect-only predictions
are exactly identical to the small-feature L1/GIoU control. Image minus aspect
IoU is +0.02034756, with family-bootstrap 95% percentile interval
[-0.06381461, +0.08959796]. Image minus permuted is +0.05537367, interval
[-0.10361279, +0.25975404]. Seeds are averaged within family before 5,000
resamples. Both intervals span zero; repeated development selection and weak
localizer targets preclude a confirmatory interpretation.

## Training-only hints

Nine matched runs add a training-only 272-parameter 16→16 pointwise adapter to
the small encoder. It matches the teacher's cached 16×4×4 rank features, divided
by fitting-only standard deviation floored at 0.1, with loss weight 0.05.
All primary targets stay attached to their true source. The no-hint adapter
is not executed and receives no gradient; a warmed-AdamW regression verifies
its inactive weights do not change. No adapter is needed at inference.

| Matched arm | Selected IoU | Selected edge MAE |
|---|---:|---:|
| No hint | 0.37429229 | 0.13616850 |
| Correct feature hint | 0.36721851 | 0.13780065 |
| Permuted feature hint | 0.37476747 | 0.13521646 |

The hint does not improve either control. Its family-bootstrap IoU difference
from no hint is -0.00707379, percentile interval [-0.01437538, -0.00008763],
on reused selection data. This is a negative pilot, not a general claim that
distillation cannot help.

The repeated baseline's epoch-zero predictions match the old end-to-end image
run exactly for all seeds. GPU training trajectories are not asserted
bit-deterministic: selected boxes differ by at most 0.00970459, 4.62924576 and
0.01038361 normalized-map pixels for seeds 31/137/911; seed 137 selects epoch
125 instead of 145. These are different trained weights, not an inference
backend parity result or a relaxation of the three-pixel inference gate.
Use matched within-study controls for the hint conclusion.

## Bounded resource evidence

Complete jobs under run-heavy, two CPU threads and CUDA per-process fraction
0.35; times include setup/evaluation/checkpoint IO, not per-step throughput:

| Job | Seconds | Peak host RSS GiB | CUDA peak allocated / reserved bytes |
|---|---:|---:|---:|
| Primary teacher, 56 images | 9 | 1.87 | 2,153,104,896 / 2,757,754,880 |
| Frozen MSE heads, nine fits | 8 | 1.44 | 18,085,888 / 23,068,672 |
| Frozen L1/GIoU heads, nine fits | 14 | 1.59 | retained in packet |
| Small end-to-end, six fits | 36 | 1.96 | 497,382,400 / 694,157,312 |
| Frozen FocalNet heads, nine fits | 16 | 1.59 | retained in packet |
| Small feature hints, nine fits | 53 | 1.95 | 490,793,984 / 750,780,416 |

Host caps are 3–5 GiB, explicit per job. No quality claim comes from run counts,
low training error or successful model loading. The required next quality gate
is source-family-independent photographic crop judgments, including acceptable
alternatives/reject-all, featured-object intent, small/distant faces and the
essential 3:1/1:3 boundaries. Existing query-derived boxes alone cannot establish
that gate.
