# Fixed extent models on broader people scenes — 2026-11-09

The earlier FocalNet extent gain on eight synthetic development scenes does not
carry to this supplementary cohort. None of the visual models beats its matched
geometry-only control. Augmentation improves the small trained encoder relative
to its unaugmented version, but does not justify promoting an extent predictor.

All 48 selected checkpoint identities and epochs were frozen before generating
new references. The [development registry](supplementary_development_2026-11-09.md)
contains 15 source families: eight newly acquired people photos, six previously
screened people/group scenes and one monkey scene. No source supplies gradients.
This coverage-selected cohort is supplementary development, not final TEST.
It has no human artistic labels or complete-face annotations.

Fifteen cached local Florence phrase-grounding calls finish with EOS, no parser
errors or token truncation, yielding 23 valid native-coordinate boxes. Targets
are unions of query results, not certified complete subjects. Queries can group
several people or omit visible people. Scores below measure overlap with these
weak references, not crop composition or detector recall.

| Fixed study, three seeds | Visual mean IoU | Geometry-only | Permuted images |
|---|---:|---:|---:|
| Small frozen, MSE |0.288559|0.362374|0.284828|
| FocalNet frozen, MSE |0.292694|0.362374|0.190489|
| Photographic frozen, MSE |0.343810|0.362374|0.296872|
| Small frozen, L1/GIoU |0.291589|0.426674|0.316683|
| Small encoder trained, L1/GIoU |0.328565|0.426674|0.353496|
| Small encoder trained with augmentation |0.382240|0.426674|0.346473|

The fit-mean-box control reaches0.360762. Each geometry comparison uses its
matched loss: the MSE and L1/GIoU controls are distinct trained models.
All three MSE geometry-control studies produce exactly equal predictions.

Across15 families, averaging three seeds before5,000 family bootstrap draws:

- FocalNet minus geometry IoU is−0.069680,95% percentile interval
  [−0.170359,+0.023231]; versus shuffled images it is+0.102205
  [+0.021832,+0.197192]. Image signal alone does not establish a useful bound.
- Augmented minus unaugmented is+0.053675 [+0.008217,+0.100807].
  Augmented minus geometry is−0.044434 [−0.099006,+0.013389].
- The augmented visual model scores0.509011 on the eight new people photos,
  0.170494 on the six older group scenes, and0.638549 on the one wildlife scene.
  These small strata are descriptive, not separate quality qualifications.

All48 numerical qualifications pass with maximum edge difference0.0 on the
retained witnesses:36 frozen heads compare freshly extracted versus cached
descriptors on the first16 original inputs;12 trained encoders replay all eight
original selected development predictions. All720 supplementary predictions
are finite, ordered and in bounds. Original fit-only normalization is reused.
No epoch, checkpoint, threshold or model selection changes.

The supplementary shuffled-input mapping uses the prior seed31/aspect-bin
recipe, moving11 of15 sources. Its mapping was bound **after** reference
generation, before parsing references or making model predictions; it was not
part of the earlier pre-reference plan. The first analysis correctly resampled
15 families but reported a hardcoded count of8. The corrected summary and the
original wrong-count file are both retained; inference was not repeated.

On i265, run-heavy reports reference generation7s, peakRSS1.99GiB,
min-available21,934MiB, peak-load3.02, with6GiB/two-thread host cap; evaluation
3s, peakRSS1.63GiB, min-available23,364MiB, peak-load2.48, with4GiB/two-thread
cap. GNU-time maxRSS is2,085,624 and1,710,760KiB respectively. CUDA allocated /
reserved peaks are648,512,000 /700,448,768 bytes for references and218,506,240 /
346,030,080 for evaluation. These are completion/resource observations, not
controlled runtime benchmarks or GPU-saturation claims.

No paid API calls, new fitting, final-TEST inference, diagnostic maps or face
certification occur. The next model investigation should test direct
aspect/purpose-conditioned crop scoring with region-retention supervision on
the broader photographic fitting set. Further training on the tiny synthetic
extent bank is unsupported by these controls. Training and runtime candidate
banks must agree; offline grounded targets must not become unavailable runtime
inputs. The original rank student and numerical/face gates remain unchanged.

See the [registered protocol](../docs/experiments/SUPPLEMENTARY_EXTENT_EVALUATION.md)
and [artifact pointer](supplementary_extents_2026-11-09.pointer.md).
