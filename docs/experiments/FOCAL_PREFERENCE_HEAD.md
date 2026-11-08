# Frozen-encoder preference-head suitability — 2026-11-08

Registered before fitting. Test whether FocalNet's pretrained crop descriptors
support the already admitted rendered-crop preferences better than the earlier
small encoder. This tests weak-label suitability, not human artistic quality.
The source initializer's historical training overlap remains unaudited.

Use only the existing 64 TRAIN-fit and 16 disjoint TRAIN-development families
in the admitted manifest (`f911af240c7a130e710222c01e7b07ac2ee4978eb2962af19d708480512456b3`).
Canonical VALIDATE/TEST images remain unopened. AI sources and local teacher
outputs supply no labels or gradients here. Source identities, candidate IDs,
labels, pairs and weights remain unchanged.

Freeze FocalNet's encoder/projection and cache its 62-value candidate descriptors
using the existing owned Lanczos letterbox/ImageNet normalization contract.
Capture the input of the original ranking head, checking descriptor replay
against its output. Save descriptors and scalar scores, not importance maps.
Initializer checkpoint SHA256:
`82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a`.

Retain the pretrained LayerNorm/62→64→32→1 rank head. Add a zero-initialized
2→64 conditioning projection for requested log aspect ratio and crop mode;
add a 32→1 acceptance head, initialized to the fit-only weighted class prior.
The conditioning starts as an additive zero, preserving initial ranking.
The image encoder remains frozen throughout.

Run three arms with the same architecture, optimizer, seeds and source order:

- Visual descriptors as captured.
- Geometry control: zero the first 51 visual descriptor values.
- Permuted control: replace those 51 values with another source's at the same
  candidate slot, separately within fit and development roles, with permutation
  seed 31. Geometry/conditioning/targets stay attached to the original source.
  Candidate slots are not guaranteed to represent identical boxes across
  sources; this control therefore tests source alignment imperfectly.

Each arm uses seeds 31/137/911, 60 epochs, batch 16, AdamW at 1e-4,
weight decay 0.01, gradient clip 5 and FP32 with TF32 disabled. Reuse the
existing equal-source BCE plus strict-pair logistic loss. Only fit indices
supply gradients. Include epoch zero in selection; choose minimum existing
TRAIN-development loss per run and retain every epoch's metrics/checkpoint.
No threshold, target, family role or face policy is tuned.

Report all seeds, selected losses, exact-bank rank acceptability and pair
accuracy, separately for 1:3–3:1 and the full bank. Compare with each arm's
initializer and the visual arm against both controls. These development
comparisons are selection diagnostics. Any bootstrap must be grouped by source
family and must disclose reuse for selection. No independent test-quality
claim or model promotion follows from a lower development loss.

Use a bounded resident descriptor cache, one explicit GPU-memory fraction and
run-heavy host-memory/thread limits. Record measured peak allocated/reserved
GPU memory and the wrapper peak RSS. No API spending is authorized by this
study. A successful fit still requires broader photographic coverage and
reviewed artistic/face annotations before shipping.
