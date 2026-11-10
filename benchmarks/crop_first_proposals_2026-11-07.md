# Direct Terra crop proposals and revised pool, 2026-11-07

The initial pilot only offered leftmost/rightmost Minimal crops. The user
rejected that framing policy. A preference between those endpoints cannot
establish a good crop. The original results remain immutable and unadmitted.

## Actual correction

Source inputs 1400/1600 and their canonical TRAIN roles are unchanged.
Terra inspected both originals and directly proposed three normalized crop
centers/scales per request: twelve proposals across four tasks. Minimal scale
is fixed at maximum area; Maximal may retain broad context or tighten.

Owned rendering combines those proposals with center/interior baselines and
multiple Maximal scales, projects centers into feasible source bounds, and
deduplicates exact source rectangles. The Rust renderer independently rejects
out-of-bounds, wrong-aspect and reduced-area Minimal proposals before writing.
The new candidate counts are 8/10/7/10: 35 crops and two original previews.
Pillow independently decodes all 37 PNGs and matches owned RGB hashes; it
writes no training/candidate pixels.

Candidate order is hash-permuted before Terra inspects all 39 image occurrences
(four original occurrences and 35 candidates). It may reject every candidate.
The result is still weak supervision with no human calibration or training
admission, and cannot certify face safety.

| Source / request | Teacher selection | Origin | Source xywh |
|---|---|---|---|
| 1400 Minimal 9:16 | C07 | Direct Terra proposal | 779,0,1944,3456 |
| 1400 Maximal 1:1 | C09 | Direct Terra proposal | 565,242,3110,3110 |
| 1600 Minimal 9:16 | C04 | Interior baseline | 466,0,1701,3024 |
| 1600 Maximal 1:1 | C09 | Direct Terra proposal | 580,16,2872,2872 |

The revised gallery puts the selected crop beside its original; all proposals
remain available for inspection. The coordinator inspected the four actual
selected PNGs. No artistic improvement percentage or detector guarantee is
inferred from these four preferences. No student was trained.

## Checks and source

Proposal generation/renderer: `8d31b2a2`; pool judgment/gallery/usage: `f4cdb052`.
Four proposal checks cover center/interior inclusion, maximum Minimal area,
off-center subject coverage, deduplication, context scales and invalid hints.
Six teacher checks retain earlier role/mapping/tie expectations and add complete
candidate assessments, invalid IDs, face-cut rejection and reject-all.

Clippy and release renderer build pass under run-heavy, 6 GiB cap/two jobs,
peak RSS 0.20 GiB. Complete rerender uses run-heavy, 4 GiB cap/one job,
peak RSS 0.11 GiB. These are tooling measurements, not inference benchmarks.

Reproduce with `just crop-first-proposal-check`,
`just crop-first-teacher-check`, `just crop-first-render-check`, and
`just crop-first-pilot-reuse MANIFESTS EXECUTABLE INPUTS NEW_OUTPUT QUARANTINE PROPOSALS`.
The direct proposal JSON and exact rendered source coordinates are retained.

## Measured teacher tokens and pricing comparison

The complete named Terra teacher session (initial capability probe, rejected
endpoint pilot, direct proposals and revised-pool judgment) recorded:

- Input: 372,720 tokens, including 323,968 cached.
- Output: 2,111 tokens; recorded cache-write tokens: zero.
- Largest individual request: 61,523 input tokens.

[Official Terra pricing](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
checked 2026-11-07, lists standard short-context rates per million tokens of
$1 uncached input, $0.05 cached input and $5 output. Applying those rates to
the recorded tokens yields **$0.0755054 API-equivalent**. The direct proposal
turn alone records 82,883 input (80,512 cached) and 660 output tokens, yielding
**$0.0096966 API-equivalent**.

These are arithmetic comparisons, not invoices. The transport is signed-in
Codex CLI; its account allowance/credit charge is unavailable here. The direct
API adapter sent no requests. Coordinator work and other sessions are excluded.
No per-image production price or large-dataset budget is extrapolated from this
pilot. `just crop-first-usage ROLLOUT OUTPUT` records completed-turn deltas,
verified rates and scope without copying private instructions/auth material.

## Retained artifacts

On i270: `/mnt/v/output/zensally/crop-first/2026-11-07-interior/`.
Mirror: `/mnt/tower/output/zensally/crop-first/2026-11-07-interior/`.
105 payload files; complete manifest SHA256:
`278fb30dd6922230d8f4faaa1275c1a45854acd1781a75e4e048635b30343c7f`.
The independent mirror and served copy both pass full membership/SHA256 checks.
Frozen source inputs retain their earlier manifest/hash and are never rewritten.
The original endpoint pilot is preserved at its separate original path.

Expanded source coverage, human teacher calibration, student training/export,
Rust scoring and production face-coverage/raster integration remain missing.
