# Candidate head-cut classifier suitability

Measured pose/gaze/scanpath inputs improve this preliminary weak classifier
relative to hidden features alone. They do not establish a useful shipping guard:
the fixed0.5 filter still cuts heads, removes many accepted crops and loses full
original-panel coverage in every run. No classifier, crop model or runtime policy
is promoted. Family uncertainty also includes zero against the permuted control.

The [registered protocol](../docs/experiments/CANDIDATE_CUT_CLASSIFIER.md)
keeps48 FIT/24 DEV and all existing candidate/label masks fixed.15 fits compare
geometry, hidden, hidden+pose, hidden+attention and permuted-attention inputs,
three seeds each. Every head is input→32 ReLU→1;32 epochs plus zero, AdamW0.001,
decay0.0001, clip5, batches8 sources/all22 queries. Normalization uses FIT
eligible features only. DEV known-class logistic loss selects epochs; routing
cut/match/availability does not. Every epoch, optimizer and RNG is retained.

Accepted-positive candidates are class0, explicit nominated-head-cut negatives
class1, and every other candidate remains unknown with zero loss/gradient.
Within each source, known classes have equal mean loss; family weights balance
sources. FIT has50367 known candidate slots on45 sources, including47408 cuts
and2959 non-cuts. DEV has24483 slots on22 sources, including23622 cuts and861
non-cuts. These nearby candidate slots are correlated, not independent photos.
Labels do not certify complete face coverage or artistic excellence.

## Aligned feature signal and uncertainty

All three aligned full-attention heads improve DEV balanced logistic loss over
the corresponding hidden and permuted heads. Seed-averaged, paired family
bootstrap over22 DEV families,2000 resamples/seed31:

| Real input minus control | Mean logistic difference | Paired95% percentile interval |
|---|---:|---|
| Full attention minus hidden |−0.0442477198719809|[−0.07994952148371588,−0.006815686249650426]|
| Full attention minus permuted |−0.041100410718089814|[−0.08415244829309151,+0.012998232451817854]|
| Full attention minus geometry |−0.0030212225392460823|[−0.07299304979811,+0.08614543412325225]|
| Pose minus hidden |−0.0246701108884405|[−0.052881505669475616,+0.004041147792965836]|

Lower is better. Seeds are averaged within family before resampling. Geometry
remains competitive on the classification objective. Epoch selection and these
analyses reuse the same development labels; this is exploratory suitability
evidence, not independent TEST confirmation or separate attribution to gaze
versus scanpaths.

## Crop availability and weak head cuts

The filter admits candidate logits below0, equivalent to cut probability below
0.5. It maximizes either fixed common endpoint sums or the existing qualified
additive-linear seed137 pair correction over surviving strict pairs. If none
survives, both endpoints are unavailable. No fallback crop or invented regret.
These two score policies do not select classifier checkpoints.

Newer essential accepted-reference panel:139 endpoints/17 families. For corrected
pair scores, unfiltered output offers139, cuts45 and matches16 references:

| Input | Offered by seeds31/137/911 | Weak cuts | Reference matches |
|---|---|---|---|
| Geometry |12/12/12|0/0/0|7/7/7|
| Hidden |90/95/95|24/27/29|14/14/15|
| Hidden+pose |100/92/94|20/20/22|15/16/15|
| Hidden+attention |99/98/103|20/21/21|18/18/18|
| Permuted attention |97/96/95|29/29/29|16/16/15|

The zero geometry cuts come with127 lost endpoint offers; they are not a safety
success. Aligned attention also loses36–41 offers. Cut reduction alone does not
measure composition or safe useful coverage. Explicit-contain and unknown-
composition cohorts are retained separately in the packet, rather than silently
removed from full-output diagnostics.

Fix a common-offer set across all six aligned/permuted checkpoints:79 endpoints
on15 families. Aligned cuts17/17/17; permuted cuts21/20/21. Aligned repairs5/3/4
and introduces1/0/0 on that same support. Seed-averaged family cut-rate difference
is−0.043756613756613764, paired95% interval[−0.09859920634920634,0.0]. This
conditional result still does not establish full139-endpoint safety.

## Original quality: now measured for every input arm

The first routing pass explicitly marks auxiliary-input old quality unmeasured.
A subsequent fixed local-teacher extension measures those inputs on all25 original
DEV sources. Reuse pinned original pose/tiled-face results and verify the complete
226-member teacher subtree. Owned resize preprocessing, gaze head padding/cap,
four six-fixation paths, feature definitions and teacher/model weights remain
unchanged.25 gaze and25 scan outputs are compact coordinates/confidence only;
internal model arrays are transient. No API calls or gradients.

Pose is present on2/25 and gaze inputs on6/25 original sources,11 head proposals
and zero omissions. The measured feature cache has30266 eligible slots and21
DEV-only donor moves under the same content-shape/hash rotation rule. Unused
candidate slots are explicit masked placeholders, not fabricated source teachers.
Cache SHA256 `848f2647921697f112f106cf13b11b016f24aa13b76b6621f5c77d4728f6eddc`.

All15 corrected guard policies lose original coverage and fail to establish the
unchanged full448-endpoint gate:

| Input | Offered original endpoints, seeds31/137/911 |
|---|---|
| Geometry |74/64/20|
| Hidden |261/258/260|
| Hidden+pose |256/252/254|
| Hidden+attention |198/204/189|
| Permuted attention |252/274/252|

Missing requests have no full-panel regret. Conditional scores remain explicitly
conditional; none is presented as a full gate pass. Unfiltered corrected output
reproduces all448 offers and original regret0.2406301498413086. The broader gate
and all original reference/quality code remain unchanged.

## Numerical replay and visual evidence

All15 selected classifier logits reproduce exactly when replayed on the cached
new DEV inputs. Separately qualify compact affine normalization and CUDA FP32
heads on all72 newer FIT/DEV sources plus25 original DEV sources with measured
auxiliary inputs, both score policies.60 comparison rows have zero eligible cut
flag changes, zero routed endpoint changes, zero availability changes, zero edge
movement and zero unchanged three-grid gate violations. Largest absolute logit
difference is3.814697265625e-6. This is cached-head/routing qualification; it
excludes the image encoder, preprocessing and teacher inference.

Three bounded diagnostic requests,17 PNG panes/14 distinct hashes; all14 directly
inspected. Owned resize/PNG output passes independent exact RGB readback. HTTP
readback verifies all17 PNGs/2625942 image bytes. No new labels are admitted.

- [Classroom6f66b42a0250bed6](https://www.flickr.com/photos/deb_and_dave/1243726188),1:1: the aligned filter shifts left and restores
  the foreground woman in Minimal. Its inner still clips the right background
  woman, already flagged by the weak mask. Repairing one endpoint is insufficient.
- [Waterfall48cd9b5fa18a01d8](https://www.flickr.com/photos/bnphotos400/7591424414),1:3: the aligned filter shifts from the seated man
  toward waterfall/right-side person and introduces a cut head fragment at the
  left of Minimal. Maximal excludes that head. The weak flag detects this failure.
- Same classroom,1:3: aligned output abstains. Unfiltered and permuted pairs
  retain boundary head cuts; missing output has text only, with no fake crop.

The source URLs above are copied from the retained witness records; those records
remain authoritative for attribution.

http://dev.lan:3300/zensally/candidate-cut-classifier/2026-11-09/witnesses/index.html

## Resources and artifact identities

i265, run-heavy, two CPU threads unless one-thread analysis;4GiB cap for fitting,
routing and numerical checks,3GiB teacher/features/rendering,2GiB preparation/analysis:

| Run | Reported duration | Peak RSS | Minimum available RAM |
|---|---:|---:|---:|
| Fifteen classifier fits |21s|1.30GiB|22111MiB|
| Initial routing |2s|1.23GiB|21991MiB|
| Old local gaze |4s|1.52GiB|21458MiB|
| Old local scanpaths |20s|1.72GiB|21458MiB|
| Original feature cache |4s|0.66GiB|22618MiB|
| Fully measured old routing |2s|1.14GiB|23086MiB|
| Compact/CUDA qualification |10s|1.76GiB|21551MiB|
| Family/common-offer analysis |0s reported|0.64GiB|23071MiB|

Teacher jobs share the GPU concurrently within their separate3GiB host caps.
These are resource observations, not serving latency, GPU saturation or deployment
memory. No compiled export or runtime speed claim follows from these failures.

Classifier packet:607 members,48956391 bytes before identity/HTTP check/manifest,
64MiB budget. Manifest SHA256
`07c10b48620eb4badf95ca815f7d67bae636d0a32282270a660f05ba7d93f3de`.
Original-input packet:114 members,47394501 bytes before identity/manifest,
96MiB budget. Manifest SHA256
`11c70e30152c020d87c1be243a93af7500eb5f5e1baa6d60ad3ac0c0bb935da9`.
On i265, roots are `/mnt/v/output/zensally/candidate-cut-classifier/2026-11-09/`
and `/mnt/v/output/zensally/original-attention-inputs/2026-11-09/`; NAS mirrors
are the same suffixes under `/mnt/tower/output/zensally/`.
Complete membership/hash readback verifies both mirrors separately.

The initial old-input preparation failed before writing outputs because the
earlier attention packet uses size-bound list manifests rather than a dictionary.
The corrected reader accepts that pinned schema and preserves the original failure
log. No source, teacher output, scientific input or quality gate is discarded.
