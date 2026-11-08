# Frozen-feature ROI head suitability

Registered 2026-11-08 after the 117-view local teacher screen meets its minimum
family support. Admit only its 53 fit-source noisy head-location targets; retain
the twelve targets in seven fixed development families for selection. They are
not artistic grades, complete head annotations or certified protection regions.
The new real-photo cohort remains outside fitting. No role changes or new API
calls are authorized by this protocol.

Freeze the target-suitability manifest and original synthetic source registry.
Keep unavailable targets unknown. Re-render each of the 65 usable ROIs directly
from original owned native RGB into an aspect-preserving 256-square letterbox,
using the pinned owned zenresize worker and zero padding. This avoids making
the already resized 768 previews training inputs. Preserve exact pixel hashes.

Use the existing pinned frozen FocalNet checkpoint, FP32 / TF32 off, ImageNet
normalization only on content and zero normalized padding. Reuse the existing
512-value descriptor: 16-channel rank projection pooled4×4, 48 fused channels
pooled2×2, global48 and zeros16. Append four ROI geometry values: log aspect,
tanh(log aspect), content width and content height. This studies localization
signal in a larger frozen encoder; it is not a fast standalone runtime model.

Reuse ExtentHead (516→64→6, ReLU and two three-way partitions) unchanged. Run
image, geometry-only and permuted-image controls for seeds31/137/911. Geometry
stays attached to the true ROI. Permute visual descriptors within fixed role
and aspect bins(<0.5,<0.9,<1.1,<2,otherwise), with seed31. Report actual moved
ROIs and source families and singleton groups; imperfect permutations remain
imperfect controls. Fit-only source-balanced mean/std normalize visual inputs,
with std floor0.001 and clipping[-10,10].

Use FP32 AdamW lr0.001, weight decay0.01, gradient norm cap5, 200 epochs and one
full 53-ROI update per epoch. Each source has equal total weight regardless of
its number of ROIs. The objective is 5× edge L1 + 2× generalized-IoU loss,
source-balanced in both fitting and development. Select minimum development
objective including epoch zero. Keep every improved checkpoint and every25th
epoch with predictions, and all scalar epoch histories. Also retain a fit-only
source-balanced mean-box baseline. No quality threshold is relaxed.

Report per-source/family edge errors and IoU, raw containment of the weak target,
all seeds and 5,000 paired family bootstrap resamples. Seven development
families are reused for selection, so intervals are exploratory. A visual gain
over both geometry/permutation controls supports further student work only;
it does not certify real faces, artistic crops or these weak teacher targets.

Bound neural inputs to65 images, batch16 encoder inference, cached head fits
under4GiB host / two-thread pool limits and 35% CUDA-memory fraction. Reserve
2GiB for fit checkpoints, keep all failures and log measured peak RSS/CUDA
allocation. Do not claim GPU saturation from this small suitability workload.
