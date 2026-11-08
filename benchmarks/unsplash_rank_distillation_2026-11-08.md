# Real-photo relative crop ranking — 2026-11-08

The small image-conditioned student improves agreement with the local FocalNet
teacher on a separately withheld photographic set. Human artistic quality,
purpose-specific suitability and complete face preservation remain unmeasured.
No model is promoted and no paid API calls were made.

## Sources and frozen roles

All144 owned sRGB working previews retain their source identities. The
[relationship audit](unsplash_relationships_2026-11-08.md) reused the owned
linear-luma fingerprint kernel: none of10,296 source pairs is within the
registered10-bit threshold. This does not certify complete duplicate recall.
Photographer components, including multi-photo components, remain together.
The deterministic registry fixes94 fit photos /81 components,25 development
/21 components, and25 TEST /22 components. Detector-positive people occur in
24/3/8 photos respectively; development people coverage is particularly small.
All photos had already been decoded and detector-screened. TEST means withheld
from this composition fitting and selection, rather than never previously seen.
Legacy/canonical corpus roles are unchanged.

## Weak supervision and matched controls

The registered cache contains2,618 fit/development queries,2,606 informative
queries and66,044 strong preference pairs. Each source has11 aspect ratios and
Minimal/Maximal modes. Targets rank an existing128-candidate bank; they are
relative local teacher preferences, not acceptable-crop annotations. Crop purpose
is unspecified. No trusted face-coverage flag is inferred from these targets.
Full/cached teacher scores match by FP32 bits on66 predeclared scout queries.

All12 fits finish: frozen image features, trainable image features, geometry
only, and permuted images, each at seeds31/137/911. The permutation moves84/94
fit and20/25 development images across families, preserving letterbox shape;
it is an imperfect control. Training uses family weighting, FP32 with TF32 off,
AdamW at0.001 with0.01 weight decay, gradient clipping5, batch16 and220 epochs.
Every fit source/query is visited exactly10 times. Unknown queries produce no
forward, optimizer or batch-normalization update. Geometry-only training leaves
the encoder, its warmed AdamW state and batch-normalization buffers unchanged.

Selection uses development family-mean teacher regret, then pair agreement,
then earlier epoch. All12 model hashes and a single export candidate were
frozen before generating TEST crop references. The candidate is trainable-image
seed911, epoch64,238,629 parameters, weights SHA256
`c002a3a796868bcd9c4c291207e3ab5d636760aa3f92249d9bf9fd58d9cde0df`.

## Independent TEST comparison

All12 frozen models are evaluated once on25 photos /22 families, including eight
detector-positive people photos. There are547 informative queries and13,986
strong pairs. Thirty-three TEST scout queries match full/cached teacher scores
by FP32 bits. The TEST permutation moves17/25 images across families.

These are family-weighted means across the three registered seeds:

| Arm | Teacher regret, lower better | Strong-pair agreement | Exact teacher top1 | Edges within3 map pixels |
|---|---:|---:|---:|---:|
| Frozen image |0.508179|81.96%|25.02%|29.81%|
| Trainable image |0.327580|86.75%|34.61%|39.89%|
| Geometry only |0.577680|77.10%|26.00%|29.25%|
| Permuted image |0.675077|76.13%|20.83%|26.24%|

Paired family-bootstrap95% intervals for trainable-image regret differences:
geometry−0.25010 [−0.42748,−0.08710], permuted−0.34750
[−0.51147,−0.20360], frozen−0.18060 [−0.27146,−0.09596]. These support useful
image-dependent ranking under this teacher. They do not measure human preference.

The single preselected seed911 has86.59% pair agreement,34.78% exact top1,
38.51% within3 map pixels, and regret0.333017. On essential ratios through3:1
and1:3 its regret is0.304571; on the secondary6:1/1:6 queries it is0.460921.
It is therefore not a crop-equivalent replacement for the teacher. Runtime
arithmetic qualification compares the same frozen student weights across
implementations, separately from this teacher-quality comparison.

## Measured training resources and profile

The complete12-fit run took179s under run-heavy: peak RSS2.02GiB,
minimum available19551MiB, peak load18.87. CUDA peak allocated584,207,360 bytes;
reserved796,917,760 bytes. These small fits do not saturate the GPU.

A separate diagnostic copy profiles200 trainable-image optimizer updates after
20 warmup updates. The fitter's math, selected models and TEST data remain
unchanged. The scoped repeat verifies memory.max4,294,967,296 bytes inside the
actual target process before fitting. It completes in42s with run-heavy peak
RSS2.85GiB, minimum available21788MiB, peak load13.17; GNU time reports
2,988,220KiB maximum RSS. The outer Nsight wrapper's0.17GiB peak is not target RSS.
Both original and scoped traces remain preserved.

The scoped capture contains272,320 kernels,1.706014984s total kernel activity
over a3.342195929s kernel span. This is an activity fraction, not SM occupancy
or hardware memory-latency evidence. Host NVTX ranges total1.380006412s for
33 development evaluations, including0.669897134s in726 query calls. CUDA API
launch calls alone occupy0.569041600s. Query graph replay and retained buffers
are the next measured optimization candidates; no speedup is claimed yet.

Protocols: [roles](../docs/experiments/UNSPLASH_ROLE_FREEZE.md),
[relative-rank targets](../docs/experiments/UNSPLASH_RANK_DISTILLATION.md).
All five finished packets have [complete NAS verification](unsplash_rank_distillation_2026-11-08.pointer.md).
