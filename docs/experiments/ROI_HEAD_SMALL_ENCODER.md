# Smaller frozen encoder control for ROI head localization

Registered 2026-11-08 after the frozen FocalNet ROI head outperforms geometry
and permuted descriptors on seven selection families. Raw containment remains
poor, so this is a soft-localization suitability experiment, never protection
certification or a shipping detector change.

Reuse exactly the65 ROI images, targets, geometry and fixed53-fit/12-development
roles from the FocalNet cache SHA256
`1b56e11b7726576842f4efc322411c526b29b9bab5902f388150cacfc10d98e3`.
Do not admit any of the101 additional real photos or144 Unsplash photos.
Keep all settings from ROI_HEAD_FIT.md: source balancing, head, loss, epochs,
seeds, normalization, three controls, selection and bootstrap reporting.

Only replace frozen visual descriptors with the owned w24 RegionCropRanker
encoder from checkpoint SHA256
`910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71`.
Pool each16-channel stride8/16 feature map to4×4, yielding512 values. Its
trained preprocessing uses RGB0–1 and zero padding; do not apply FocalNet's
ImageNet normalization. This encoder/preprocessing comparison is not a pure
parameter-count ablation. No encoder gradients or hint losses are introduced.

Retain the source cache identity and image hashes; check geometry, target and
source metadata equality before fitting. Geometry-only per-epoch metrics must
match the original run exactly; a difference requires diagnosis, not a weaker
expectation. Record actual encoder parameters and serialized encoder-state
bytes separately from deployment dependencies and process RSS. No runtime
latency is inferred from parameter count or cache time.

Cache batch16 under4GiB host/two-thread and35% CUDA-memory bounds; repeat the
same nine fits with independent outputs. Preserve checkpoints and failures.
The same small development set selects epochs; results remain exploratory.
