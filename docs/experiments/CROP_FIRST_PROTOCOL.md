# Crop-first composition experiment

User requirements, 2026-11-07: artistic composition, requested aspect ratio,
minimal/maximal options, minimal cropping only when necessary, never crop a face.
This protocol precedes inspection or training on any newly acquired crop labels.
The current implementation is an untrained research prototype, not a new model
release or a changed shipping detector contract.

## Behavior and supervision

- Minimal: keep the source unchanged when no ratio change is requested. If a
  ratio forces cropping, rank placements only at the largest feasible size.
- Maximal: rank tighter compositions at several scales, preserving every
  protected face. Tightness alone is not the composition objective.
- All known faces are hard constraints, not confidence/area-weighted subjects.
  Unknown face coverage, no legal frame, or no finite score returns contain.
  Padding changes the output canvas; it must not distort the source.
- Face coverage is a trusted annotation/caller assertion. Empty detections or
  a VLM's own confidence do not prove that every face was found. Missed faces,
  profiles, tiny/group faces and uncertainty need independent audits. A claim
  about known protected regions is not universal face-detection accuracy.
- Continuous source-space boxes precede rasterization. A production renderer
  must conservatively preserve protected bounds in its actual pixel footprint.
  No shipping rasterizer integration is claimed by this prototype.

## Architecture

Reuse the owned w24 convolutional encoder, omit the saliency decoder, and learn
crop scores directly. Encode the aspect-fitted source once, then score many
ratio/mode queries from the cached features. The initial scorer combines:

1. A 3x3 grid of features inside each candidate.
2. Features from four disjoint discarded bands and global content context.
3. Candidate coordinates/area, source and target aspect, mode and face geometry.

A small MLP ranks alternatives. Hard legal masks and face-union containment
are checked outside learned weights and again at selection. Multiple good
answers remain possible; a single averaged regression box need not represent
their composition. The first candidate generator samples five scales; neither
search optimality nor tight-zoom quality has been established.

A letterboxed thumbnail preserves geometry but can lose detail on extreme
aspects. The thin-content test establishes finite arithmetic only. Train/evaluate
aspect-covering overlapping views and candidate-local refinement as separate
ablations before claiming extreme-aspect composition quality; those additions
are not implemented in this student prototype.

Training uses preferences within the same source, target ratio and mode.
Different requests cannot supply a pairwise comparison. Human candidate scores
or preferences anchor composition; Terra judgments are separately tagged weak
labels. Ties, neither and uncertainty remain explicit. No self-reported model
confidence is treated as a calibrated probability.

## Dataset admission

The historical pilot's PNG-v3 roles do not establish compatibility with the
current JXL-sRGB split. The 2026-11-08 original-hash audit finds pilot TRAIN
6080 in current VALIDATE 16080. See [the reconciliation finding](../../DATA_PROVENANCE.md#current-jxl-srgb-split-reconciliation-required-2026-11-08).
Reconcile current source families and historical evaluation exposure in a new
admission manifest before fitting; preserve the frozen pilot and its labels.

Existing imazen-26 roles stay fixed. All crops, sizes and judgments inherit
their parent source role. Existing validation images, DUTS-TE and the earlier
crop galleries cannot become training data. Quarantine train members of known
cross-role source families; do not move their validation/test relatives into
training. The family index is not a complete perceptual-duplicate audit.

Build our own vision-supervised crop-preference data first. The user selected
this direction after the external-source survey. No GNMC archive was downloaded.
External human crop datasets remain possible calibration references, subject to
separate provenance/terms/split admission:

| Source | What it can supply | Admission status |
|---|---|---|
| [GAIC](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) | Multiple scored candidate crops; ranking supervision | Publisher links checked; archive/terms/splits still need admission |
| [GNMC](https://zenodo.org/records/6228834) | Editor crops at five named ratios | Metadata only; published PolyForm Noncommercial terms exclude automatic commercial-training admission |
| [FCDB](https://github.com/yiling-chen/flickr-cropping-dataset) | Human composition/crop annotations | Publisher source identified; bytes and terms not admitted |
| [UGCrop5K/S2CNet](https://github.com/suyukun666/S2CNet) | More complex UGC composition/content cases | Paper/code verified; actual UGC dataset download not established |

The GNMC README's stated total and split counts are inconsistent; measure
archive membership before reporting usable counts. Preserve official roles,
all source URLs/hashes and raw annotations. No new archive is silently merged
with the training pool. No GAN/outpainted panorama substitutes for real
extreme-aspect originals in the final evaluation.

Broaden the source pool with real panoramas, portraits/groups, edge faces,
products, food and multi-object scenes. Keep screenshots/documents separate:
photo aesthetics do not establish permission to discard text. Audit overlap
across datasets before source-family splitting, including perceptual duplicates.

## Vision-teacher protocol

[Cropper](https://arxiv.org/abs/2408.07790) demonstrates VLM-based cropping,
including requested aspect ratios. That supports testing a teacher, not treating
its judgments as human truth. [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
supports image inputs and structured output; the account's actual access must
be probed before a large run.

For each admitted train source and requested ratio/mode, present the original
and two actually rendered, anonymously named candidates. Randomize order and
repeat swapped comparisons on a calibration subset. Store input hashes,
canonical roles, candidate mapping, prompt/schema hashes, exact model/date,
raw response, usage and admission state. Preserve failed and refused requests.
Do not ask a VLM to hallucinate pixel coordinates and then call them accurate.

The teacher may rank composition, identify destructive cuts and reject both.
It cannot certify full face coverage. An eventual annotation process must
calibrate teacher preferences against human judgments, assess order bias and
disagreement, and route uncertain cases to review. Existing infrastructure
must run any batch; the adapter makes one explicit request and has no hidden
retries or autonomous paid sweep.

The first CLI probe uses the already-inspected reported 00084919 case, with
one blank and one subject-retaining crop. It is evaluation-only and a capability
check, not an artistic benchmark, order-bias study or training example. No
student may train on this probe. A direct API key is not configured on the
current host; the signed-in CLI path is a separate authenticated transport.

## Scientific gates and missing work

Freeze source groups and a human-rated evaluation set before fitting. Human
test preferences must remain independent of teacher/model selection. Compare
against center crop, corrected saliency placement and simple face-preserving
baselines. Evaluate both modes, multiple ratios, real source aspect groups and
content classes. Report ties, abstentions, unnecessary-padding rate, legal-crop
coverage, area removed and face violations, not just average model preference.

Known protected-face violations must be zero. An all-padding policy cannot
claim success from that gate alone. Composition preference should improve on
the frozen baseline with source-clustered uncertainty; report test sample sizes
and all failed strata. Never use mask retention as the artistic gold label.

Still missing: admitted expanded training labels, human teacher calibration,
student training, trained-model quality measurements, export, a Rust scorer,
production face-coverage/rasterization integration, latency/allocation profiling
and real extreme-aspect human evaluation. No speed or deployment-size target
is claimed as measured. Start with the existing training optimizer, then compare
choices on validation only; do not tune on the new human test set.

## Geometry teachers: SAM and Detectron2

[Meta SAM 3](https://github.com/facebookresearch/sam3) supports concept-prompted
object detection/segmentation. Use it offline to propose subjects and refine
masks; use a dedicated face/head detector plus review for protected bounds.
Do not require a whole person mask to survive: a deliberate portrait may cut
legs while every known face remains protected. Subject-mask preservation is a
composition feature, not the face-safety contract.

The [SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
primarily improves multiplexed video-object tracking. Its name does not establish
better still-image crop judgments. [Detectron2](https://github.com/facebookresearch/detectron2)
is a detection/segmentation framework; a selected checkpoint supplies predictions.
Neither provides a calibrated artistic crop score or proves complete face coverage.
No SAM/Detectron checkpoint was downloaded or evaluated in this experiment.

The intended division is geometry teacher + vision-model composition preferences
+ a small crop-first student. Large teachers stay offline; serving the student
need not add SAM, Detectron2, PyTorch or an LLM to the Rust dependency graph.

## Own-source pilot

The initial two-source pilot forced a choice between leftmost/rightmost Minimal
crops and two centered Maximal scales. The user rejected that proposal policy:
a teacher's preference within such a pair cannot establish the best composition.
Its immutable results remain historical and unadmitted.

The revised pilot asks Terra directly for three source-conditioned crop centers
and scales per request, then renders them with owned Rust pixels. Maximum-area
Minimal center/interior baselines and broad/tighter Maximal baselines join the
pool. Exact source rectangles are deduplicated; source bounds, exact aspect and
maximum integer Minimal area are validated before rendering. Vision centers are
projected into legal image bounds, then actual rendered pixels are inspected;
coordinates alone never establish quality.

On frozen TRAIN sources 1400/1600 the revised four pools contain 8/10/7/10
candidates: 35 crops and two originals. Presentation order is hash-permuted.
Terra inspects all rendered alternatives and can select none. The gallery shows
the selected crop beside its original, with every proposal available for review.
The current preference selects a direct vision proposal for three tasks and an
interior baseline for one. Neither preference nor its rationale is human truth.

`just crop-first-pilot-reuse` verifies frozen source hashes and leaves inputs
unchanged. The complete pilot retains `unreviewed`, `training_eligible=false`
and `face_safety_certified=false`. No model has trained on these labels. The
[revised report](../../benchmarks/crop_first_proposals_2026-11-07.md) records
actual crops, independent readback, teacher tokens and API-equivalent pricing.


## Operations and optimizer

The encoder uses existing owned convolution/depthwise and squeeze/excitation
operations. The new head requires bilinear feature gathers, weighted pooling,
concatenation, small dense layers and ReLU. Research uses PyTorch `grid_sample`;
export and Rust implementation of the gathers/head remain missing.

The existing `training/train_v3ds.py` and `train_v3ds32.py` use AdamW with
weight decay 1e-4. Start with AdamW and within-task Bradley–Terry preference
loss, then compare alternatives on validation. This is a proposal: no crop
student has been trained, and no optimizer comparison has been measured.
Candidate generation and hard face containment use bounded geometry, not a
learned optimizer. Ties/rejections need explicit loss/admission treatment;
the initial pairwise loss accepts only admitted strict preferences.

## Expanded ratio study (2026-11-07–08)

The expanded frozen study covers 32 imazen-26 sources (24 TRAIN, 8 VALIDATE),
all 21 content groups and 448 source/ratio/mode requests. Ratios are 1:1, 9:16,
16:9, 1:4, 4:1, 1:8 and 8:1. Known cross-role TRAIN relatives are excluded;
no TEST source is used. Roles and admission remain unchanged. This is source
coverage, not a calibrated artistic evaluation. Only one source exceeds aspect
4, and it is a screenshot; natural extreme panoramas remain missing.

Each main pool contains the center control and up to two direct Terra proposals,
deduplicated and rendered before judging. A separate four-source, 16-request
audit broadens extreme-ratio placement coverage without changing crop geometry
or face criteria. Its judgments must remain separate from the main 448.

Preview rendering uses an isolated published zenresize 0.3.1 worker, owned PNG
decode/encode and strided RGB input. It does not add a runtime dependency to the
shipping crates. A matched one-image probe measured 403 fewer total input tokens
at 768×576 than at 1024×768; it does not establish crop/face accuracy at either
size. See the [token measurements](../../benchmarks/crop_preview_tokens_2026-11-07.md).

[Astra's consultation](CROP_NETWORK_ASTRA_2026-11-07.md) recommends projected
stride-8/16 encoder features, fractional region pooling of retained/discarded
content and crop boundaries, plus ranking and independent acceptability heads.
Aspect-preserving rectangular plans and optional cached overlapping views need
ablations. Those changes are proposals; the current student still uses point
samples. A separate pooled research variant is implemented below. Neither has
been fitted to admitted artistic labels or exported to a Rust crop runtime.

The main teacher selects 86/448, with zero selections at 1:8/8:1. The separate
audit selects 4/16, all on one abstract source, while also reversing rejection
of identical center controls. No label is promoted from this sensitivity check.
See the [expanded results](../../benchmarks/crop_expanded_2026-11-08.md).

## Accelerated training suitability (2026-11-08)

`training/crop_regions.py` now implements an experimental projected stride-8/16
head with nine retained bins, four discarded bands, eight boundary strips and
global context. Separate logits rank crops and predict acceptability. Fractional
pooling uses a summed-area table and bilinear corner interpolation; independent
cell-overlap values/gradients, empty bands, thin content, cached chunks and hard
face masking are checked. The original point-sampling model remains available.

`just crop-train-probe` exercises synthetic end-to-end gradient/optimizer updates
with bounded shapes, eager/compiled/graph paths, precision/layout choices and
GPU telemetry. Host memory is capped externally by `run-heavy`; the CUDA tensor
allocator is capped at 70% of device memory. Allocated, reserved and sampled
whole-device usage have different meanings and must remain separately labelled.
No synthetic overlap objective is an artistic composition quality measurement.
Existing unreviewed corpus judgments remain training-ineligible unless the owner
explicitly admits TRAIN-only noisy supervision. No held-out role may change.

The [preliminary training-path report](../../benchmarks/crop_training_2026-11-08.md)
records the actual GPU/CPU scouts, profiler evidence, memory counters and limits.
Synthetic target loss must not be reported as composition quality.

## Direct API coordinate supervision and response caching

The owner selected direct API calls using configured dev credentials. One
image-attached, tool-free Responses call supplies ROI boxes and all ratio/mode
crop zones together in original-source percentages. Four TRAIN-only sources
produce 18 proposed ROIs, 56 decisions and 44 rendered alternatives. The
[direct API report](../../benchmarks/crop_direct_api_2026-11-08.md) retains the
measured usage, percentages, pixel checks and limitations.

Cache complete requests/responses by canonical request hash, including image
bytes and model/prompt/schema/settings. Repeated identical requests verify
stored hashes and require zero API calls. Retain completed raw responses even
when downstream annotations are invalid; incomplete/failed attempts remain
separate. Reparse saved bytes when fixing a parser rather than resending.
Full-source contain zones preserve all 0–100% bounds; coordinate proposals
still require rendering and review. No caller gains face-coverage certification
or training admission from a VLM's ROI/face estimates.
