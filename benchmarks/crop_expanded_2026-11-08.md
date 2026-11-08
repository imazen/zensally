# Expanded crop-composition study

Measured 2026-11-07–08. This is an unreviewed vision-teacher study, not human
artistic quality, face-detection accuracy or trained-model performance.

The frozen selection contains 32 imazen-26 sources: 24 TRAIN and 8 VALIDATE,
all 21 content groups, no TEST source and no known quarantined cross-role TRAIN
relative. Each source has seven target ratios and Minimal/Maximal modes:
448 requests. Canonical roles and training admission remain unchanged.

## Teacher decisions

| Target | Selected / judged | Rejected all |
|---|---:|---:|
| 1:1 | 34 / 64 | 30 |
| 9:16 | 26 / 64 | 38 |
| 16:9 | 18 / 64 | 46 |
| 1:4 | 2 / 64 | 62 |
| 4:1 | 6 / 64 | 58 |
| 1:8 | 0 / 64 | 64 |
| 8:1 | 0 / 64 | 64 |
| All | 86 / 448 | 362 |

Minimal and Maximal each select 43/224. TRAIN selects 72/336 and VALIDATE
14/112; those are teacher choices, not independent accuracy. Of the 86 winners,
54 are distinct direct vision proposals, 20 coincide with center controls and
12 are center-only controls. No accepted candidate has a teacher face-cut
assessment of yes or uncertain; this is not a face-safety certificate.

The main pool contains center plus up to two direct proposals. A contain
proposal leaves only the center negative control. Therefore rejection does not
establish that no good crop exists. A separate four-source/16-request audit
expands placements and scales at unchanged geometry/face criteria. Its results
are recorded separately in the artifact summary.

The companion audit completes 16/16 requests, rendering/checking 160 crop
rectangles and 164 distinct PNGs. It selects four crops, all on source 1556
(abstract ice), and rejects the other twelve. Two winners are interior controls
and two are added axis controls. All four positive judgments also accept the
**identical center-control PNGs** rejected in the main study; their hashes match
across runs. Candidate pools and conversation context/order differ, so this is
evidence of teacher sensitivity, not a controlled causal placement improvement.
Those labels must not be treated as an artistic gold standard.

There are 26 sources with aspect magnitude below 2, five from 2 to below 4,
and one at least 4. The last is a screenshot. Real extreme natural panoramas,
human preference calibration, order-bias calibration and student training remain
missing. Screenshots/documents use an explicit meaningful-text preservation rule;
their rejections cannot stand in for photographic aesthetics.

## Pixels and preview tokens

All 32 downloaded PNGs match pinned Git/LFS objects at PNG-v3 revision
`b1faae550eba4d29efee1de1a32954ef11a0e1ed`. Owned decoded RGB SHA256 hashes
match an independent read-only Pillow decode for all 32. Two stale manifest
URLs resolve through pinned tracked inventory, preserving both URL/dimension
records without changing corpus manifests or source identities.

All 587 main candidate rectangles pass source bounds, exact integer aspect
and maximum integer Minimal-area checks. All 619 distinct source/candidate PNG
hashes match annotation inputs. Chromium checks all 448 selector combinations,
source-bounded overlays and 619 image decodes with no JavaScript exception.

The isolated published zenresize 0.3.1 worker renders native strided RGB.
Packed/strided, batch/individual and independent crop-origin byte checks pass.
It adds no shipping runtime dependency. Source previews use a 768-pixel long
edge; candidate previews have a 262,144-pixel and 1,536-pixel-long-edge cap.

Three matched fresh calls on one image measured total input tokens of 31,991
at 1024×768, 31,588 at 768×576, and 31,300 at 512×384. Thus the adopted 768
preview reduced this probe by 403 tokens. Image-only metering and crop/face
quality at these sizes are unmeasured. [Full probe](crop_preview_tokens_2026-11-07.md).

## Design consultation

The actual read-only **gpt-6-astra/high** consultation recommends retaining the
shared w24 encoder initially, projecting stride-8/16 features, and using
fractional area pooling for 3×3 retained bins, discarded bands, boundary strips
and global context. A small ranker and separate acceptability head handle
relative composition and rejection separately. Keep known-face geometry outside
learned scores; unknown coverage or infeasible framing returns contain/pad.

Compare aspect-preserving rectangular input plans and cached source-dependent
overlapping views as explicit ablations. Start with AdamW/cosine, within-task
Bradley–Terry preferences and separately calibrated acceptability; no optimizer
or latency advantage has been measured. Existing convolution/depthwise/pointwise,
SE and dense operations are reusable; pooling/search/export/output-buffer
integration remain work. [Full consultation](../docs/experiments/CROP_NETWORK_ASTRA_2026-11-07.md).

The current 261,637-parameter student remains untrained, with point samples,
no exported crop weights, Rust scorer, serving-size or crop-latency measurement.
The consultation has not silently changed that implementation.

## Execution evidence

All heavyweight jobs use `run-heavy`; no native-CPU build flags. Preview build:
13 s, peak RSS 0.34 GiB (6 GiB cap, two jobs). Source acquisition/rendering:
32 s, peak 0.20 GiB (4 GiB, one job). Full RGB oracle: 7 s, peak 0.36 GiB.
Main Chromium verification: 2 s, peak 0.19 GiB (4 GiB, two jobs).

The finite resumed supervisor completed in 1,370 s, peak RSS 0.11 GiB, with
durable per-batch stage/failure records. An earlier 60-second turn timeout was
operationally too short; its completed model response was captured without
reissuing the call, and the supervisor deadline became 300 seconds. No scientific
gate or annotation criterion changed. The failed attempt and logs are retained.

Streaming CLI evidence reduced peak RSS on the same settled log from 0.69 to
0.10 GiB with byte-identical captured proposals/evidence. This concerns annotation
tooling, not network inference. The first companion-audit call returned a provider
Bad Request without labels; its failure is retained rather than admitted as data.

Reproduction entry points are `just crop-expanded-plan/run`,
`crop-expanded-report`, `crop-coverage-report`, `crop-expanded-gallery` and
`crop-gallery-browser`. Full raw artifacts retain exact command logs, proposal
and judgment prompts/responses, usage, images, source roles, binaries/locks,
pixel oracles and membership/SHA manifests. Private full CLI rollouts are excluded.

Final Chromium verification includes both galleries: 448/16 requests, 619/164
image decodes, all selectors and previous/next/next-selected navigation, with
no JavaScript exceptions. Wrapper: 2 s, peak RSS 0.19 GiB (4 GiB, two jobs).
Companion rendering/readback: 3 s, peak RSS 0.14 GiB (4 GiB, one job).

## Recorded usage and API comparison

| Scope | Recorded standard API equivalent USD |
|---|---:|
| Main expanded proposals/judgments, 32 turns | 2.0713896 |
| Companion audit, including one failed turn | 0.6007365 |
| Astra consultation | 1.284578 |
| Three matched preview probes | 0.0606628 |
| This expansion/consultation/probes | 4.0173669 |
| Earlier teacher pilot in the reused session | 0.0755054 |
| All named sessions combined | 4.0928723 |

Rates verified 2026-11-08 from official [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
and [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) pages.
The calculation applies per-request long-context adjustments and recorded cache
hits. The reused Terra session records 21,832,297 input tokens (21,221,504 cached)
and 54,290 output tokens. These totals include earlier pilot work. Long CLI
context and tool round trips are part of the measured annotation transport.

This is **not an invoice or actual subscription credit charge**. Actual billing,
allowance and any unreported failed-call usage are unknown. Coordinator/other
sessions are excluded. No future per-image price is extrapolated. Stateless
source-level batches with attached images are a proposed way to reduce context
overhead; savings have not been measured. `just crop-cost-summary` reconciles
the exact named-session ledgers and explicit task scopes.
