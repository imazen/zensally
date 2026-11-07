# Saliency aspect ratios, tiling and crop refinement

**Historical solver results:** the user identified a crop-placement defect
in `ILSVRC2012_test_00084919 / canvas-1x8-p0`: the map localizes the upper
subject while the 9:16 crop excludes it. The original measurements below
are preserved for that solver. Crop comparisons and failure attribution are
superseded by the [corrected placement and blur study](w24_crop_placement_blur_2026-11-07.md).
Mask MAE is unaffected. The corrected solver raises natural stretch retention
from 71.435% to 80.354% and changes the extreme overlap gain from 0.066 to
1.987 percentage points.

**Original interpretation (superseded where noted above):** keep square
stretching as the default for the current w24 weights. Black
letterboxing and four independently evaluated quadrants regress on natural
photos. Overlapping refinement improves mask error on controlled extreme
canvases, but its crop improvement is small. A calibration-selected policy
has no clear natural-image crop gain and requires more model passes.

The interactive gallery has 29 cases: selected improvements/regressions,
aspect-stratified examples and three unlabelled real wide/tall sources.
It shows full-image heatmaps, exact crop rectangles and actual extracted crops.

http://dev.lan:3300/zensally/aspect/2026-11-07/visuals/index.html

## Natural images

These are **3,760 evaluation images**, separate from 1,259 calibration images,
from the existing 5,019-pair DUTS-TE archive. Foreground is human annotated.
The common scoring grid is 256x256; probability threshold is 0.5. MAE is
lower-is-better. Crop retention is higher-is-better.

| Method | Model passes | Mask MAE | Minimal foreground retained | Minimal retention regret |
|---|---:|---:|---:|---:|
| Stretch | 1 | 0.08058 | 71.435% | 13.258% |
| Black letterbox | 1 | 0.11415 | 70.026% | 14.664% |
| Mean-color letterbox | 1 | 0.11130 | 69.459% | 15.232% |
| Four stretched quadrants | 4 | 0.14449 | 64.384% | 20.307% |
| Global + four selected overlapping views | 5 | 0.08786 | 71.347% | 13.346% |
| Global + aspect-covering views | 3.039 average | 0.08052 | 71.417% | 13.276% |
| Calibration-selected policy | 2.794 average | 0.08063 | 71.445% | 13.249% |

Minimal is the shared solver's largest crop at the requested ratio. Regret
is the difference between captured foreground and the best feasible placement
of a crop with that same size, measured on the scoring grid. It accounts for
foreground loss that the requested crop size makes unavoidable. Targets are
1:1, 16:9, 9:16, 4:1 and 1:4; Minimal/Maximal modes are both recorded.

The selected policy's Minimal regret difference versus stretch is
-0.0000917; the paired source-bootstrap 95% interval is
[-0.0009900, 0.0007769]. That does not confirm an improvement. Selected
four-view refinement slightly increases natural-image regret: +0.0008856,
interval [0.0002386, 0.0015745]. Its extra work is not justified as a default.
Intervals are paired estimates, not a familywise hypothesis-testing procedure.

Maximal mode retains 75.335% of foreground at 47.786% mean source area for
stretch; aspect-covering refinement retains 75.511% at 47.641%; selected
four-view refinement retains 75.599% at 47.639%. These small shifts are not
human preference scores. The complete typed tables preserve every rectangle.

### A concrete stretch failure

The first gallery case, `sun_ahfgmrgdocgbpcnm`, contains a small person near
the right edge of a 400x300 snowy scene. For a 1:1 Minimal crop, stretch
chooses `(50,0,300,300)` and retains 1.928% of the foreground mask. Letterbox
chooses `(100,0,300,300)` and retains 100%. Both crops are valid; the mask
oracle confirms full retention is feasible. This example was deliberately
selected as a large improvement, not as a representative average.

The gallery also contains large regressions from letterboxing and tiling.
Individual stretch failures do not establish that aspect preservation is a
better default for these weights. The training loader explicitly resizes
RGB and masks to squares (`training/train_v3ds.py`). Letterboxing changes
content scale, introduces padding and changes the input distribution. The
experiment does not isolate geometric distortion from those other effects.

## Controlled extreme aspect ratios

There are 64 calibration and 64 evaluation source photos, each rendered into
21 canvases: seven aspects (1:1, 2:1, 4:1, 8:1 and reciprocals), three
positions. Source proportions are preserved. A photo is fitted inside a
120-pixel maximum footprint on a 128-pixel short-axis canvas; the background
is the mean of the original four corner colors. The mask is transformed with
the image. This is synthetic background extension, not a natural-panorama
benchmark. All variants inherit their source partition.

The **4:1/8:1 and reciprocal evaluation group** has 768 cases from 64 sources.

| Method | Model passes | Mask MAE | Minimal foreground retained | Minimal regret |
|---|---:|---:|---:|---:|
| Stretch | 1 | 0.09020 | 83.596% | 11.591% |
| Black letterbox | 1 | 0.30924 | 52.367% | 42.820% |
| Mean-color letterbox | 1 | 0.10938 | 82.181% | 13.007% |
| Four stretched quadrants | 4 | 0.11018 | 55.871% | 39.316% |
| Global + four overlapping views | 5 | 0.06015 | 83.662% | 11.526% |
| Global + aspect-covering views | 12 average | 0.06664 | 83.623% | 11.564% |

Selected overlapping refinement reduces MAE by 0.0300431, paired 95% interval
[-0.0324850, -0.0275461]. Minimal regret falls by only 0.0006552, interval
[-0.0012601, -0.0000726]: about **0.066 percentage points** of foreground.
The improvement in the probability map is much larger than the improvement
in crop placement. Background suppression contributes to MAE; this experiment
does not prove that the extra calls primarily recover finer boundaries.
No separate probability-calibration-only ablation was run.

Aspect-covering refinement reduces MAE but has no clear Minimal crop gain:
regret difference -0.0002706, interval [-0.0018549, 0.0020266]. Its extra views
are not an established improvement over the cheaper selected-view method.
These pass counts describe complete model evaluations, not measured latency;
no per-pass timing extrapolation or speed claim is made.

## Implemented refinement behavior

The model remains a fixed 256x256 executor. Every local prediction is mapped
back into source coordinates and accumulated with tapered overlap weights.
A whole-image prediction remains available everywhere, including uncovered
pixels. The experiment does not independently normalize each tile's scores.

The selected-view method generates overlapping square candidates of side
5/8 of the source short side, scores candidates from global foreground and
uncertainty, penalizes overlap with views already selected, and chooses up to
four. The aspect-covering method uses full-short-axis square views, covers
the long axis with at least 50% overlap, and caps the covering ensemble at
16 views. Above that budget it enlarges rectangular windows and letterboxes
them. Tests cover translated coordinates, strided crop rows, odd sizes,
overlap accumulation and gap-free bounded coverage.

Fusion uses a fixed, unfitted global gate:

`result = global + 0.5 * global * (weighted_local - global)`

Where no view contributes, the global value is retained. A zero global value
cannot be promoted by this gate, so this is not a guaranteed rescue path for
subjects the global model completely misses.

The two channel-attention blocks compute spatial means, and the bottleneck
uses dilated convolutions. Independent image tiles change scene context and
introduce new borders; they do not reproduce one larger full-image evaluation.
The losses measured for quadrants are consistent with context/scale changes,
but the contribution of each mechanism was not isolated.
[Global/local saliency research](https://arxiv.org/abs/1908.07274) supports
combining scene context with local detail; its trained fusion model is not
implemented or claimed by this experiment.

## Real extreme cases and practical handling

DUTS-TE here contains **no natural source with aspect ratio >=4**. The gallery
therefore separately renders three original, unlabelled imazen-26 validation
sources: ID 8023 (1080x5192 screenshot), 3313 (900x239 wide image), and 8003
(1080x2520 screenshot). Original URLs and hashes remain in `real-inputs.tsv`.
These are visual-review cases; there are no fabricated foreground scores.
All three sources are opaque, including the palette PNG.

For the current weights, retain stretch as the one-pass default and avoid
stitching independent quadrant maps. The implemented global-plus-overlap
path is an opt-in experiment for extreme geometry/detail, with a measured
cost in model passes and only a small crop gain on controlled canvases.
Do not promise its synthetic results on natural panoramas or documents.

A 1:1 crop of the 1080x5192 screenshot can keep at most 1080 of its 5192 rows
(20.8% of source area). Higher-resolution saliency cannot make that crop retain
all article text. Where text completeness matters, use contain/padding or
explicit focus constraints instead of treating photo-object saliency as a
measure of paragraph importance. No document-intent model was evaluated.

## Source, selection and uncertainty

Protocol was committed before quality inspection (36306336). Calibration-only
selection and bootstrap rules were frozen in 2460bee9. Complete inference
uses 5dac6169; the separately frozen renderer uses 0fa21350; analysis is
1e17cef0. The NN/model values remain unchanged. The shared solver is pinned
to zenpipe 841ab33f, including its bounds correction and published-feature
compatibility. Exact executable, source and input identities are retained.

For each aspect bin [1,1.5), [1.5,2), [2,4), [4,infinity), the selection rule
chooses the calibration method with lowest mean Minimal regret. Ties within
1e-6 prefer fewer passes, then method name. Bins with fewer than 30 calibration
sources fall back to stretch. Natural and synthetic calibration stay separate.
No per-image mask enters ROI selection, prediction or fusion.

Every source retains its role. No weights were trained. Historical architecture
selection already inspected DUTS-TE, so this is not a pristine architecture
holdout. One exact-byte duplicate pair exists within evaluation, with no
cross-partition byte overlap. Point estimates retain all benchmark rows;
bootstrap uncertainty resamples identical-byte sources together, with their
row weights. Natural evaluation has 3,759 such clusters; stress evaluation
has 64. Near duplicates were not audited. Both uncertainty packets are kept;
point estimates and calibration choices are exactly unchanged by this correction.

DUTS source archive SHA256:
`d9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104`.
Frozen 5019-pair manifest SHA256:
`beeea44d84dac1106df0772c81f8f869862dd5e1176690b9a0c1b5b6db005df3`.
Source: [DUTS dataset](https://saliencydetection.net/duts/). The old model
exploration numbers use a different evaluation implementation; these controlled
preprocessing comparisons do not replace that historical benchmark table.

## Validation and corrected defects

The complete study has 5,019 natural cases and 2,688 stress cases, six methods,
ten crops per method: **462,420 labelled crop comparisons**. Every scored crop
has positive dimensions and lies inside its source. Mask/native image dimension
checks, source hashes and completion markers are mandatory. Foreground-mask
and crop-oracle calculations use the common 256x256 grid; thin features and
subpixel boundaries remain discretization limits. Mask retention is a useful
proxy, not a human aesthetic crop rating.

The original shared Maximal solver independently rounded a half-pixel origin
and extent: a 400x300 source returned x=38, width=363. The fix constrains the
integer origin while retaining the rounded size. A synthetic horizontal and
vertical regression preserves the 363-pixel extent and requires x/y=37.
All-feature targets, alloc-only tests, doctests and Clippy pass for the isolated
zenlayout crate. Its published `alloc` feature was also restored for compatible
consumer selection. No old numerical assertion was weakened.

The initial complete run failed on that bounds check. A subsequent run exposed
RGB noise in a few ground-truth masks. The reader now uses the training loader's
BT.601 grayscale convention followed by the declared threshold. All frozen
bytes and all 5,019 pairs are kept; both partial runs remain failed evidence.
Pillow independently verifies **all 5,019 decoded native binary-mask hashes**.

Gallery verification checks all 156 selected labelled method rows against the
full run. All **1,273 generated PNGs** pass independent pixel-hash readback;
13 natural mask grids also match independent coordinate/threshold sampling.
A captured renderer reproduces all 1,273 PNG files byte for byte. Chromium
loads all 29 cases and checks crop/aspect/opacity controls with no JavaScript
exceptions. An initial CDP shutdown-handler error is retained with the
corrected successful browser run. Public API snapshot checks pass.

run-heavy wrappers report: complete evaluation 145 s / 0.02 GiB peak RSS;
source-clustered analysis 8 s / 0.34 GiB; independent gallery/mask oracle
12 s / 0.05 GiB; renderer reproduction 19 s / 0.05 GiB; API check
43 s / 0.82 GiB. These are observed wrapper summaries, not aggregate live
heap or deployment latency measurements. The four evaluation children share
a hard 6 GiB cgroup cap. No native CPU target flags were used.

[Compact results](w24_aspect_crops_2026-11-07.json) preserve primary statistics
and hashes. Full Parquet+zstd tables, raw shard outputs, failed attempts,
executable captures, browser evidence and a verified independent mirror are
listed in [the artifact pointer](w24_aspect_crops_2026-11-07.pointer.md).
