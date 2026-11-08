# Frozen-encoder crop preferences — 2026-11-08

Neither fit establishes better photographic composition. Mixed-corpus weak
development scores improve, but the photographic subset does not; fitting only
the photographic subset reduces its ranking quality. No checkpoint is promoted.
Canonical VALIDATE/TEST images remain unopened throughout these studies.

## Inputs and registered comparisons

The [mixed-cohort protocol](../docs/experiments/FOCAL_PREFERENCE_HEAD.md) uses
the unchanged 64 TRAIN-fit / 16 TRAIN-development families and their existing
rendered-crop labels. It adds no local teacher or AI composition labels.
The frozen FocalNet checkpoint is
`82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a`;
its original training inventory/overlap remains unaudited.

The 80×128×62 descriptor cache has SHA256
`8d4095679480b52799a7ee55dc06460497795b1301590d149cb1b908aa99eefa`.
Every cached descriptor exactly replays the original Torch rank output.
Owned Lanczos letterboxing and the existing ImageNet/content normalization
contract remain unchanged. No importance or metric diagnostic maps are saved.

The trainable head has **6,430 parameters**, retaining pretrained LayerNorm
and 62→64→32→1 ranking layers, plus zero-initialized aspect/mode conditioning
and a separate acceptance output. Its initial rank scores match the frozen
head exactly on all 80 sources. Encoder/projection weights stay frozen.
The acceptance prior uses fit sources only.

Each visual, zero-visual geometry and role-local source-permuted arm uses
seeds 31/137/911, AdamW 1e-4, weight decay 0.01, clip 5, FP32/TF32-off and
the existing equal-source acceptance BCE plus strict-pair logistic loss.
Epoch zero participates in selection. All epochs/checkpoints are retained.
Permuted candidate slots need not denote identical boxes across sources;
that control imperfectly tests source alignment.

## Mixed-corpus development results

All three visual runs select epoch 60, as do both controls. Means across seeds:

| Head | Combined loss | Pair accuracy | Accepted top ranks / judged tasks |
|---|---:|---:|---:|
| Frozen visual initializer | 1.0340934 | 65.9704% | 119/338 |
| Fitted visual | 0.9184109 | 70.4731% | 134–135/337 |
| Geometry control | 1.0287497 | 68.7116% | 104/339 |
| Permuted visual control | 1.0335160 | 66.8763% | 122–124/338 |

There are 352 development tasks, with 13 of 16 families supplying strict pairs.
For essential ratios (1:3 through 3:1), the visual initializer has 110/274
accepted top ranks; fitted visual has 121–123/273. Unknown/excluded labels
are omitted from the judged denominator, not counted as rejection.

Selected checkpoints recover their saved choices/actions and development
losses. A paired 5,000-resample family bootstrap averages seeds within families
first. Visual minus initializer pair accuracy is +4.5027 percentage points,
percentile interval [+0.1968, +9.2468]. Visual minus geometry is +1.7615 points
[-18.8005, +17.4608]; versus permuted it is +3.5968 points
[-5.6914, +12.7863]. The same families select checkpoints: these are exploratory
selection diagnostics, not independent confidence statements about art quality.

Only **five** development sources meet the already established photographic
metadata scope. The posthoc photographic subset has visual pair accuracy
58.0393% before fitting and 57.3284% afterward; geometry reaches 69.2710%.
The aggregate gain must not stand in for the requested photographic task.

## Photographic-only suitability

The separately registered [photographic-cohort protocol](../docs/experiments/PHOTO_PREFERENCE_HEAD.md)
uses the existing category policy to select **15 fit / five development**
sources. No family moves roles; all other cached sources supply neither
gradients nor checkpoint selection. There are three people-category fit sources
and one people-category development source. This does not cover the desired
range of group scenes, distant people, products or ecommerce contexts.

All arms retain the same architecture/optimizer. One 15-source update per
epoch for 240 epochs matches the mixed study's 240 optimizer updates; cohort,
source reuse and batch shape change together, so this is not a causal filtering
comparison. Visual selects epochs 237/217/216; both controls select 240.

| Head | Combined loss | Pair accuracy | Accepted top ranks / judged tasks |
|---|---:|---:|---:|
| Frozen visual initializer | 1.2995967 | 58.0393% | 64/108 |
| Fitted visual | 1.2673094 | 54.1696% | 57–58/107 |
| Geometry control | 1.2795437 | 68.8031% | 55/109 |
| Permuted visual control | 1.4228800 | 48.7120% | 59/107 |

There are 110 tasks, including 90 essential-ratio tasks. Essential accepted
top ranks fall from 56/88 to 49–50/87. The visual-minus-initializer paired
family bootstrap gives pair accuracy -3.8697 points [-7.7471, -0.0955] and
essential accepted-rank fraction -6.5482 points [-12.1909, -2.2928]. Five
families, weak labels and reuse for selection limit every inference.

The visual acceptance loss falls from 0.6910567 to 0.6661031; rank loss falls
from 0.6085401 to 0.6012063, while pair accuracy/top-rank acceptance worsen.
Lower combined loss therefore does not establish a better crop selector.
The study does not isolate whether shared gradients, calibration, labels or
sample coverage cause that mismatch.

## Actual pixels and containment

Eight existing rendered images across three source families are inspected
without new image generation or label admission. On portrait 12008, both raw
3:1 and 1:3 choices visibly cut the face and remain withheld. On mirror selfie
11068, the photo-fit 1:3 raw candidate C03 shifts away from the central reflected
person compared with the initial C01; the fitted action is contain. On 11460,
the wide flower crop retains a portion of the tall flower: a weak acceptable
label is not a complete-subject guarantee. These are assistant observations,
not human ratings or certified face annotations.

The comparison gallery uses the existing owned-rendered pixels, shows full
sources for withheld choices by default and offers an explicit raw-rank toggle.
Weak labels start hidden. User reviews have a separate storage namespace and
cannot automatically enter training. Browser checks pass 16 extreme-ratio,
amount and raw/contain combinations, decoding 90 distinct images without
stretching or JavaScript exceptions. Only five TRAIN-development families are
included; the gallery introduces no canonical holdout exposure.

http://dev.lan:3300/zensally/composition-photo-preferences/2026-11-08/gallery/index.html

## Resource evidence and limits

All jobs use run-heavy with explicit host memory/thread limits and a 35% GPU
allocator fraction on RTX 4080 / Core Ultra 7 265K. No paid API calls are made.

| Job | Elapsed s | Wrapper peak RSS GiB | Peak CUDA allocated / reserved bytes |
|---|---:|---:|---:|
| Frozen descriptor extraction | 3 | 1.52 | 34,250,752 / 39,845,888 |
| Nine mixed fits | 15 | 1.40 | 32,567,808 / 48,234,496 |
| Nine photographic fits | 25 | 1.40 | 25,499,136 / 46,137,344 |
| Mixed checkpoint recovery/bootstrap | 1 | 1.09 | not recorded |
| Photographic checkpoint recovery/bootstrap | 1 | 1.10 | not recorded |
| Browser check | 1 | 0.19 | not applicable |

Descriptor extraction / mixed fits / photo fits have minimum available RAM
24,088 / 24,176 / 24,191 MiB and peak loads 12.17 / 12.29 / 12.79. These short
small-head jobs are not GPU-saturation or matched training-throughput claims.
Raw logs and checkpoints are retained. The head's initialization/conditioning
gradient unit test passes; existing training contracts are preserved.

The native timing in the [cached-runtime report](cached_crop_runtime_2026-11-08.md)
uses the **unchanged pretrained head**, not either newly fitted head. Neither
fit has a Rust export, native numerical gate, calibrated shipping abstention
policy or face certification. Broader photographic sources and reviewed crop
judgments remain necessary; further selection on five families cannot replace
them.
