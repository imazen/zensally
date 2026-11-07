# Corrected crop placement and blurred letterboxing

The user was right: the original gallery's stretch 9:16 crop for
`ILSVRC2012_test_00084919 / canvas-1x8-p0` discarded a subject that the map
localized. The shared solver's hot-region bounding box included distant weak
speckles; its final adjustment moved the crop off the dominant upper band.
This was a crop-placement defect, not evidence that stretching failed to
localize that subject. The original study is preserved and its crop
interpretation is superseded by this complete corrected-solver comparison.

## Reported case

The source is a controlled 128x1024 canvas. Minimal 9:16 changes from
`(0,637,128,227)` to `(0,60,128,227)`; foreground retention changes from 0% to
100%. Maximal changes from `(0,398,128,228)` to `(0,60,128,228)`, also 0% to
100%. Input, prediction and crop dimensions are unchanged.

The matched 29-case gallery exposes original solver outlines in cyan,
corrected outlines in yellow, actual extracted crops and the exact padded
256x256 model inputs. Whole-map fit avoids hiding the context on tall sources;
turn it off to zoom and scroll. Every case supports direct links.

http://dev.lan:3300/zensally/aspect-corrected/2026-11-07/visuals/index.html?id=ILSVRC2012_test_00084919&aspect=9%3A16&mode=minimal

## Placement repair

Both heatmap-only modes now maximize retained squared saliency mass at their
already chosen crop size. Source pixels and fractional heatmap-cell overlap
are accounted for. Equal-mass plateaus retain the previous composition.
Explicit focus-region rules are unchanged. The candidate grid covers integer
maxima of the piecewise bilinear box integral; sliding vertical strips and a
horizontal prefix use scratch linear in map width/height, without a 2D table.
There are four temporary scratch vectors per crop; no allocation or latency
improvement is claimed for this correctness repair.

A synthetic regression reproduces the old y=637 failure in both orientations
and both modes. An independent exhaustive source-pixel oracle checks fractional
cell boundaries. Existing expectations, including the earlier rounded-extent
regression, remain unchanged. Std/alloc-only tests, doctests, Clippy and the
full workspace API snapshot check pass. No public API was added.

An independent Python oracle reads selected f32 maps, projects squared mass by
rows and exhausts every integer vertical origin for all 80 crops of the reported
case: eight maps and 66,456 origin comparisons. Every corrected crop reaches
the best mass at its fixed size. This checks placement against the prediction;
it does not turn saliency into a guarantee of human aesthetic preference.

## Natural images

All 5,019 DUTS-TE sources are retained: 1,259 calibration, 3,760 evaluation.
The table uses the evaluation partition, five target ratios and Minimal mode.
The same fixed weights, frozen roles and source-byte clusters are used.

| Method | Passes | Mask MAE | Minimal foreground retained |
|---|---:|---:|---:|
| stretch | 1.000 | 0.08058 | 80.354% |
| letterbox | 1.000 | 0.11415 | 79.018% |
| letterbox-mean | 1.000 | 0.11130 | 78.960% |
| letterbox-blur | 1.000 | 0.10265 | 79.356% |
| letterbox-edge-blur | 1.000 | 0.10404 | 79.285% |
| quadrants | 4.000 | 0.14449 | 69.740% |
| overlap-refine | 5.000 | 0.08786 | 80.432% |
| aspect-refine | 3.039 | 0.08052 | 80.495% |

For stretch, correcting placement raises mean Minimal retention from 71.435%
to 80.354%, with the same mask MAE. Maximal retention rises from 75.335% to
80.319%, at unchanged crop sizes for every original comparison.

Aspect-covering refinement has a small natural-image gain: Minimal regret
change -0.0014270, paired source-bootstrap 95% interval
[-0.0024268, -0.0004355]. The calibration-selected policy gives retention
80.499% at 3.025 average model passes. Its regret change is -0.0014680,
interval [-0.0024843, -0.0005083]. The six-arm and eight-arm policies select
identical methods here. Five-pass selected overlap has no clear natural crop
gain: regret change -0.0007846, interval [-0.0019967, 0.0004168]. These are
nominal paired intervals without multiple-comparison correction.

Retain one-pass stretch as the practical default for the current weights;
aspect refinement is an optional small crop-quality tradeoff for extra calls.
Independent quadrants remain substantially worse. Mask errors and crop
retention answer different questions; the earlier solver obscured that distinction.

## Fixed blurred-background arms

Both new arms use one model pass, the same aspect-fitted foreground and the
same output unpadding. A single separable radius-16 box blur runs on the
256x256 background. Running sums are adapted from owned zenmetrics a3329089,
`crates/margarine/box_blur.rs`; no new dependency is added. The foreground is
restored bit for bit after preparing the background.

- `letterbox-blur`: center-cover the square background from the source,
  blur it, then composite fitted content.
- `letterbox-edge-blur`: extend fitted edge pixels into padding, blur the
  aligned background, then restore fitted content.

Blurred padding improves on black letterboxing on natural photos, but both
arms still lose to stretch. Natural Minimal regret versus stretch increases
by 0.0099535 for cover blur (95% interval [0.0076968, 0.0123440]) and
0.0106669 for edge blur ([0.0084815, 0.0130490]). Neither arm is selected by
calibration. Radius was frozen before evaluation; no radius or blend was fitted.

The method does not eliminate all artificial gradients: restoring sharp content
over a blurred background still creates a transition. Cover backgrounds can
repeat unrelated interior objects; edge extension avoids that duplication.
The gallery shows the actual inputs so these boundaries are inspectable.
No seam-gradient benchmark or radius sweep establishes an optimum here.

## Controlled extreme aspects

The 4:1/8:1 and reciprocal evaluation group has 768 cases from 64 source
photos. These are synthetic constant-background extensions, not natural
panoramas; variants inherit their source partition.

| Method | Passes | Mask MAE | Minimal foreground retained |
|---|---:|---:|---:|
| stretch | 1.000 | 0.09020 | 86.955% |
| letterbox | 1.000 | 0.30924 | 64.930% |
| letterbox-mean | 1.000 | 0.10938 | 85.605% |
| letterbox-blur | 1.000 | 0.12732 | 84.619% |
| letterbox-edge-blur | 1.000 | 0.10922 | 85.912% |
| quadrants | 4.000 | 0.11018 | 75.821% |
| overlap-refine | 5.000 | 0.06015 | 88.942% |
| aspect-refine | 12.000 | 0.06664 | 87.563% |

Global plus four overlapping views now improves Minimal retention by
1.987 percentage points over stretch. Its regret change is -0.0198736,
paired 95% interval [-0.0318952, -0.0078142]. The earlier 0.066-point crop
gain described the faulty solver and is superseded. At five model calls this
supports optional refinement for extreme/detail-sensitive cases, with the
same global gate's inability to recover a subject whose global value is zero.

DUTS-TE contains no natural source with aspect ratio >=4. Three real opaque
wide/tall sources remain unlabelled gallery cases. No quality claim on real
panoramas or document intent follows from this synthetic test. For long
text that must survive, contain/padding or explicit focus constraints remain
necessary: the target crop itself may discard most source area.

## Reproduction and evidence

Inference source: 95c7a8ca6314; shared crop solver: zenpipe 6ed32dea4df1.
The frozen extension protocol is in
[ASPECT_CROPS_PROTOCOL.md](../docs/experiments/ASPECT_CROPS_PROTOCOL.md).
Architecture selection historically saw DUTS-TE: this is not a pristine
architecture holdout. No source bytes, masks, roles or model weights changed.
The bootstrap uses 3,759 independent byte clusters for 3,760 natural evaluation
images and 64 source clusters for stress. Near duplicates remain unaudited.

All 46,242 original reported prediction rows match in MAE, IoU, F-beta,
source dimensions and model-call count. All 462,420 original crop sizes match;
421,194 crop origins change. All 174 original gallery heatmap PNGs are byte
identical, including the real cases. Full-precision original maps were not
captured for the original full run; metric equality is not claimed as a
full-corpus bitwise map comparison.

The complete eight-arm rerun has 616,560 source-bounded labelled crops.
Independent readback verifies all 5,019 native binary masks and 1,887 PNGs,
including the separately rendered reported case; 13 natural mask grids match
independent mapping. The gallery matches all 208 selected labelled method rows
against full inference. Chromium loads all 29 cases/eight cards, verifies
reported coordinates, original overlays, links, controls and image/heatmap
alignment, with zero JavaScript exceptions.

Observed run-heavy summaries: full evaluation 356 s / 0.02 GiB peak RSS;
strict old/new comparison 5 s / 0.15 GiB; analysis 14 s / 0.41 GiB;
independent PNG/mask oracle 18 s / 0.03 GiB; shared API check
8 s / 0.44 GiB. Wrapper RSS is not aggregate simultaneous live heap or a
latency benchmark. Evaluation used four workers under a hard 6 GiB cap.
The added arms and changed placement work prevent interpreting the run duration
as a paired runtime speed comparison. No native CPU target flags were used.

[Compact results](w24_crop_placement_blur_2026-11-07.json) and
[artifact pointer](w24_crop_placement_blur_2026-11-07.pointer.md) preserve all
methods, original failures, paired results, source identities and mirror hashes.
