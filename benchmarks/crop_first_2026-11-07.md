# Crop-first prototype and own-source labeling pilot, 2026-11-07

Missing: expanded admitted labels, human teacher calibration, student training,
export/Rust scorer, production face-coverage/raster integration, crop quality
and latency measurements. This is an untrained research prototype.

## Implemented and checked

Research implementation: `fbc881f4`; owned renderer/pilot: `114e44ef`.
No new normal Rust dependency, model weights or shipping crop API was added.
The current shipping detector/solver is not replaced by these Python guards.

The encoder is shared across candidate/ratio/mode queries. Actual parameter
counts: 163,332 encoder + 98,305 scorer = 261,637 total. The full-resolution
smoke encodes `[1,3,256,256]` once to `[1,72,16,16]`, then scores 472 legal
candidates across ten ratio/mode tasks. All scores are finite and all candidate
rectangles contain the supplied synthetic face. This is numerical/architecture
coverage, not learned artistic quality or detection accuracy.

Seven policy checks and seven model checks pass: all known faces, unknown
coverage/contain, infeasible ratios, adversarial unsafe scores, cached scoring,
permutation, thin content, gradients and rejected cross-task preferences.
Four teacher-admission checks pass: held-out role exclusion, malformed/refused
responses, order mapping, ties and no automatic label/face certification.

Commands: `just crop-first-policy-check`, `just crop-first-model-check`,
`just crop-first-smoke OUTPUT`, `just crop-first-teacher-check`.
The combined policy/model/smoke/inventory run used run-heavy, 4 GiB cap/two
jobs: peak RSS 0.29 GiB, elapsed 6 s. Teacher checks: peak RSS 0.02 GiB.
These are test-process measurements, not deployed runtime memory/latency.

## Corpus audit

Canonical PNG-v3 manifest checkout: `b1faae550eba4d29efee1de1a32954ef11a0e1ed`.
The inspected train/validate/test/family manifests have no local changes;
an unrelated representatives manifest is locally modified and was not used.
TRAIN manifest SHA256:
`5be8ebe18e46e066b946f94a5759581ca8915a20e4008d4e16142ba7740630ac`.

| Canonical role | Sources | Long/short aspect <2 | [2,4) | At least 4 |
|---|---:|---:|---:|---:|
| Train | 1084 | 1062 | 22 | 0 |
| Validate | 658 | 644 | 13 | 1 |
| Test | 418 | 406 | 11 | 1 |

Dimensions above are manifest observations. The known family index identifies
90 cross-role families and 162 TRAIN sources to quarantine, without moving
any canonical role. This is not a complete perceptual-duplicate audit.
The two pilot sources are outside that known quarantine.

## Actual vision judgments

An evaluation-only signed-in Terra CLI probe chose the subject-retaining
00084919 crop over a blank alternative. That obvious comparison is not an
artistic benchmark and never becomes training data.

The bounded own-source pilot then supplies four comparisons to
`gpt-6.1-terra`, with twelve inspected image occurrences / ten unique previews.
Actual rendered pixels and source-space crop boxes accompany every label:

| Source | Request | Teacher preference | Teacher accepts A / B |
|---|---|---|---|
| 1400 | Minimal 9:16 | B | false / true |
| 1400 | Maximal 1:1 | A | true / true |
| 1600 | Minimal 9:16 | B | true / true |
| 1600 | Maximal 1:1 | A | true / false |

The teacher discusses corner distractions, valley context and whole dumplings;
these rationales remain unreviewed weak labels. No human agreement, order-bias
calibration or complete face coverage is measured. The proposals deliberately
include unfiltered annotation examples and are never crop recommendations.
All four records remain `training_eligible=false`, `admission=unreviewed`,
`face_safety_certified=false`. No student has trained on them.

The signed-in CLI pilot recorded 106,242 input tokens (96,640 cached) and 520
output tokens across its tool-assisted turn. This includes repeated CLI context;
it is not a direct API token/cost measurement. A direct Responses request was
prepared offline with the actual images. The API transport was not called:
`OPENAI_API_KEY` is absent. No SAM/Detectron weights or external crop archive
was downloaded.

## Rendering and retained evidence

`just crop-first-render-check` passes Clippy and a release build: run-heavy
6 GiB cap/two jobs, peak RSS 0.19 GiB. Complete source acquisition/rendering
uses owned Rust PNG pixels and bounded candidates; run-heavy 4 GiB cap/one job,
peak RSS 0.11 GiB. Pillow independently decodes all ten output previews and
matches their owned RGB hashes; it writes no candidate/training pixels.

Input root on i270: `/mnt/v/input/zensally-crop-first-2026-11-07/`.
Mirror: `/mnt/tower/input/zensally-crop-first-2026-11-07/`.
Four payload files; manifest SHA256:
`2891649d4e8c7a1b3562834ffb137eae5bb0090f992c9f957d5491449f517133`.
Original-source hashes and downloaded PNG hashes are retained separately.

Output root on i270: `/mnt/v/output/zensally/crop-first/2026-11-07/`.
Mirror: `/mnt/tower/output/zensally/crop-first/2026-11-07/`.
52 payload files; manifest SHA256:
`0b3d911419889c5c7bc2cadc9786461662789f26ce9cd4ff575f647b3db7fb04`.
Both mirrors and the served gallery copy pass full membership/SHA256 checks.
Frozen renderer, relevant source, logs, source/candidate hashes, weak labels,
prompt and usage are retained. Private CLI instructions/auth context are not
copied. HTTP readback matches the HTML and all twelve linked assets/metadata.

The [protocol](../docs/experiments/CROP_FIRST_PROTOCOL.md) owns model choices,
hard constraints, data roles, optimizer proposal and outstanding gates.
