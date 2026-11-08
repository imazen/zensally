# Current-corpus weak crop fitting (2026-11-08)

The current-JXL pilot now fits real rendered weak labels. Quality at the
1:3/3:1 boundaries is still insufficient for adoption. No human artistic
ratings, complete face-coverage audit, Rust export or shipping model is claimed.
The earlier PNG-v3 pilot and its labels remain historical and excluded.

## Frozen sources and supervision

The pinned [current export](https://github.com/imazen/imazen-26-jxl-srgb)
uses `1` plus the original four-digit source ID and the successor
[family map](https://github.com/imazen/imazen-26/blob/32ba0ebea7b3cccbd6e141a397d90423140a7fc6/manifests/split_map_family_v2.tsv).
The freeze excludes 66 current TRAIN families with historical held-out members.
This is the registered family map, not a complete perceptual-duplicate audit.
The frozen export manifests have zero cross-role groups sharing an encoded,
original-file or declared pixel SHA256. Selected sources also have zero
cross-use matches in owned decoded RGB or fitted-input hashes.

| Use | Canonical role | Distinct source families | Usable candidate labels | Strict ranking pairs |
|---|---|---:|---:|---:|
| Fit | TRAIN | 64 | 4,840 | 2,040 |
| Internal development | TRAIN | 16 | 1,135 | 441 |
| Evaluation only | VALIDATE | 16 | 1,174 | 451 |
| Held back, unopened | TEST | 8 | 0 | 0 |

Only Fit supplies gradients. Internal development selects the checkpoint.
Canonical validation is evaluated after selection; TEST bytes are never opened.
Crops, previews and labels inherit parent roles. Candidate counts are correlated
views of these source families, not independent photographs.

All 96 non-TEST sources are actual hashed JXL-sRGB bytes. Owned decode, zenresize
and PNG encode produce fitted 256-square model inputs and annotation previews.
All 7,416 rendered PNGs agree in owned and independent read-only decoded RGB;
one lossless JXL also matches libjxl exactly. Original links, licenses and
attribution remain in source packets, the gallery and the admission-provenance
sidecar. No third-party reader writes model inputs.

Terra proposes ROIs and 22 ratio/mode decisions in one call per source. Exact
integer crops and interior baselines then supply 7,320 rendered candidates for
384 separate pixel-review requests. One invalid containment response is retained;
a corrected request passes the unchanged full-source containment rule.
Completed responses are immutable, content-addressed and separate from admission.

Sol blindly reviews the same extreme-ratio pixels on 18 TRAIN sources (36 calls,
144 tasks). Acceptability agreement is 227/250 at 1:3/3:1 and 220/248 at 1:6/6:1.
Preferred-choice agreement is 51/72 and 50/72 respectively. Audited disagreements,
uncertain face assessments and positive labels conflicting with proposed face
geometry are excluded without reversing their labels. Reject-all tasks train
acceptability without inventing ranking winners. Essential tasks have unit
weight; 1:6/6:1 tasks have weight 0.25.

The 517 unique completed model responses, including the invalid annotation,
use 1,396,227 input and 607,416 output tokens. The native usage ledger computes
USD 5.07389025 at the checked [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
and [Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) Standard rates,
including reported cache writes. This is a published-rate calculation, not a
verified invoice. Every call uses the configured dev API credential; repeated
requests reuse the local response cache.

## Fitting and frozen evaluation

The existing 238,726-parameter region scorer receives source-balanced ranking
and acceptability losses. Weak ROIs do not set the serving scorer's trusted
face-coverage flag: raw training logits and the independent serving guard remain
separate. The trained checkpoints are research artifacts, not deployable crops.

Two 100-epoch, compiled BF16/channels-last/fused-AdamW fits use seed 31 and batch
16. The saliency initializer transfers the existing owned encoder, excluding its
decoder/auxiliary heads; its checkpoint SHA256 is
`8019bebd6557db31435edf393b0da034d2eb2975a485f74f906a02d8e04ed1b7`.

| Initialization | Best internal-development epoch | Best loss | Final epoch-100 loss | run-heavy peak RSS |
|---|---:|---:|---:|---:|
| Fresh weights | 14 | 1.0481659174 | 2.3395290375 | 2.66 GiB |
| Existing saliency encoder | 11 | 0.9312002063 | 1.5956468582 | 2.64 GiB |

Both overfit after early checkpoints. Selection is frozen before canonical
evaluation: initialized encoder, epoch 11, checkpoint SHA256
`910219ef1af5eecf4530b675fecddba04798b62e323def2b781de8fda5354f71`.
The training manifest SHA256 is
`f911af240c7a130e710222c01e7b07ac2ee4978eb2962af19d708480512456b3`.

The fixed research policy takes the top raw rank only if its acceptability
probability is at least 0.5 and it contains all proposed face ROIs. Otherwise it
retains the full image. The threshold is not fitted to canonical validation.
Trusted face coverage is still unverified, so this is a diagnostic policy;
the serving guard would continue to fail closed for these unreviewed sources.

| Canonical evaluation tasks | Suggested crops / requests | Teacher acceptable / judged suggestions |
|---|---:|---:|
| All nine essential ratios, 1:3 through 3:1 | 115 / 288 | 90 / 112 (3 unjudged) |
| 3:1 boundary | 5 / 32 | 3 / 5 |
| 1:3 boundary | 9 / 32 | 6 / 9 |
| 6:1 secondary | 7 / 32 | 6 / 7 |
| 1:6 secondary | 8 / 32 | 3 / 8 |

The pooled result hides weak boundary behavior and substantial abstention.
These are Terra-label diagnostics on 16 sources, not human quality estimates.
A visual spot check confirms the rejected 13323 3:1 painting crop cuts through
flower heads; the accepted 11483 3:1 landscape keeps the mountain peak/context.
Those evaluation observations do not change training inputs or select a model.

## Training-path resources and remaining limit

Candidate context is packed once. Streaming loaders retain workers and bound
image caches to 64 MiB per worker. An optional device-resident backend checks a
shared payload budget before allocation and preserves pixel/context values and
the epoch sampler's source order. CPU fixtures recover identical model state
and history after interruption/resume through both backends. Every epoch's
checkpoint payload is retained; `best.pth`/`last.pth` are movable pointers.

The first real-input batch-64 trace averages 26.17% SM activity, 0.71% Tensor
activity and 29.20% kernel busy time. The 64 MiB resident-cache trace records
90.10% kernel busy time, with 50,760,960 bytes of resident input/context payload
and CUDA peak allocated/reserved bytes 1,026,033,152 / 1,333,788,672. The dominant
named GPU kernel is fractional-pooling `grid_sampler_2d_backward` in the first
trace. A separate real-input CPU process under sudo perf reports 33.5%
memory-bound and 36.0% core-bound slots; run-heavy peak RSS is 1.05 GiB.
These counters include the scopes recorded in their artifacts, not a claim that
all training stages or Tensor units are saturated.

The root-run Nsight capture uses system scratch despite both overrides; the
user-run capture honors the configured scratch path and omits hardware metrics.
Final reports are retained. Nsight's run-heavy wrapper RSS is not the target
Python process's RSS. The initial profiling helper also consumed its loader's
global RNG before model construction; the frozen paired helper corrects this.
The two traces are diagnostic evidence, not a controlled speed comparison.

A complete additional 100-epoch fit exercises the resident backend: best internal
loss 0.9354494810 at epoch 11; CUDA peak reserved 478,150,656 bytes; run-heavy peak
RSS 2.69 GiB. Its weights are not bit-identical to the streaming run, and are not
promoted or used to repeat canonical evaluation. Source-order/pixel parity does
not assert deterministic CUDA gradient reductions or crop-quality equivalence.

The zenbench four-block A/B/A-A plan is retained, but a verified live benchmark
admission prevents it from starting. There is no valid paired speed result and
no statistical speedup claim. Preserve the admission and rerun the retained plan
only when the scheduler admits it.

Human artistic calibration, verified face coverage and stronger boundary-ratio
quality remain missing. The review gallery supports source overlays, mode/ratio
filters and local human voting/export. No TEST exposure or shipping crop change
is needed to review these preliminary findings.

Reproduction entry points: `just crop-corpus-freeze`, `crop-current-acquire`,
`crop-fit-plan`, `crop-fit-render`, `crop-fit-api`, `crop-fit-admit`, `crop-fit`,
`crop-fit-profile`, `crop-fit-policy-report`, `crop-api-ledger`, `crop-fit-gallery`,
`crop-fit-browser`, `crop-fit-check` and `crop-fit-resume-check`. See the
[artifact pointer](crop_fitting_2026-11-08.pointer.md) for frozen evidence.
