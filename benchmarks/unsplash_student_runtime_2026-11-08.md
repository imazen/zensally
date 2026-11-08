# Selected photographic student runtime — 2026-11-08

Human artistic quality and complete face preservation remain unmeasured. This
study qualifies arithmetic for the pre-TEST-selected relative-rank student;
it does not promote a crop model or certify teacher-equivalent rectangles.
[Training/quality findings](unsplash_rank_distillation_2026-11-08.md) remain separate.

## Export and crop gate

The selected238,629-parameter seed911/epoch64 student exports as static batch1
FP32 ONNX opset18, including all exported weights:

| Graph | Bytes | Nodes |
|---|---:|---:|
| Encoder |667,688|75|
| Query |339,949|448|
| Complete |1,007,616|523|

These are graph sizes, not executable/dependency sizes. ONNX Runtime remains
required for this experiment. The existing native FocalNet query benchmark
does not measure this newly trained student's head.

All144 sources ×22 ratio/mode requests are compared against the same frozen
student on CUDA, FP32 with TF32 disabled, batch1. The unchanged gate allows
at most3 source-normalized128-grid pixels on every edge independently.

| Runtime | Requests | Changed choices | Gate violations | Maximum edge movement | Maximum absolute score error |
|---|---:|---:|---:|---:|---:|
| Torch CPU |3,168|1|0|0.068966|0.000726522|
| ORT cached encoder/query |3,168|3|0|0.017106|0.000600577|
| ORT complete graph |3,168|3|0|0.017106|0.000600577|

Coordinates normalize x and y by their respective source dimensions. This
comparison never uses teacher ranking scores, updates weights or reselects a
model. It does not yet compare batch1 with the original batched evaluation.
All face/acceptance guards remain separate; raw ranking alone cannot authorize
an offered crop.

The first export wrapper accidentally restored training mode recursively after
export, invalidating its Torch CPU reference. That attempt is retained. Fresh
wrappers now enter evaluation mode explicitly, and export must preserve every
parameter, batch-normalization buffer and evaluation flag. The corrected full
gate takes41s: run-heavy peak RSS1.40GiB, minimum available20489MiB,
peak load13.81. CUDA peak allocated18,293,760 bytes/reserved27,262,976 bytes.

## Validation launch reduction

A private CUDA graph replays only the fixed-shape candidate query. Each
evaluation copies current scorer parameters into retained storage and refreshes
feature buffers once. The original encoder, masking, nonfinite checks,
reductions, CPU metrics and prediction construction remain unchanged.
Every selected model is checked, including changed head parameters and
geometry/permutation controls:844,800 FP32 scores match by bits. All12 original
metric/prediction records, model state and RNG state remain unchanged.
This qualification takes2s: run-heavy peak RSS1.56GiB,
minimum available23460MiB, peak load13.04. CUDA allocated269,370,368 bytes,
reserved408,944,640 bytes.

Four paired exploratory zenbench blocks on i265, CPU boundary2–3 and RTX4080,
measure the whole25-photo/22-query development evaluation at30.2563ms eager
and23.0328ms with query replay, a23.8737% reduction. A/A change is+0.6524%.
Twelve invocations record background CPU load above5%; this is not a clean-resource
confirmation. CUDA event intervals include submission gaps and explicit end
synchronization. Model/cache loading and graph capture/warmup are excluded.
The run-heavy wrapper reports33s, peak RSS1.57GiB, minimum available23408MiB,
peak load12.54; its outer accounting does not replace per-target memory telemetry.
No complete training-speed improvement is claimed from validation timing alone.
