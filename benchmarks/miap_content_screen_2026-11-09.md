# Broader photographic people screen — 2026-11-09

All 384 fixed fitting/development sources now have original JPEGs, owned previews,
independent RGB readback, cached local content hints, relationship screening and
source-by-source visual observations. No artistic crop labels, face certificates
or new training gradients are admitted by this screen. The 96 TEST sources remain
unopened. See [immutable packets](miap_content_screen_2026-11-09.pointer.md).

## Acquisition and numerical checks

The [group-aware pilot](miap_group_pilot_2026-11-09.md) supplies 288 fitting and
96 development images, one per frozen source component. Original bytes total
124,443,136 bytes and retain download URLs, HTTP metadata and provider identities.
The source records contain 2,518 person boxes, including group records; this is
not a count of individual people. Demographic fields are unused.

Producer 369b28e3 uses the pinned owned JPEG/PNG pipeline and PNG reader, strict
sRGB and Lanczos3 previews at at most 768 pixels on the long edge. All 384
previews decode independently to the exact stored RGB bytes. Twelve earlier
witnesses also match the refactored helper's complete row objects and PNG/RGB
hashes exactly. Strided pixel contracts and prior numerical expectations remain.

An initial RGB-only attempt stopped at its first grayscale JPEG after 36 completed
records. Those records and the next original JPEG were reused without deletion or
reinterpretation. Explicit grayscale admission accepts native RGB or L; the
default RGB-only guard remains. Final coverage is 369 RGB and 15 L sources. An
additional grayscale witness confirms equal channels and exact independent RGB
readback, including the owned encoder's palette PNG representation.

Reproduce with `just composition-acquire-miap-pilot`, the corrected pilot,
annotation and twelve-witness packets, pinned pipeline/reader, retained helper
parity JSON, `--allow-grayscale` and `--previous-cache` pointing to the preserved
RGB-only attempt. Exact arguments and GNU time output are in the packet logs.
The corrected run takes 217 seconds under 2 GiB/one thread; run-heavy reports
peak RSS **0.06 GiB**, minimum available RAM 20,598 MiB and peak load 6.30.
The original failed attempt takes 24 seconds with peak RSS 0.04 GiB.

## Cached hints and independent review

Native Florence-2-base at pinned revision
00921df66db728a9ceb750f5eca43e5c203a2051 supplies `<MORE_DETAILED_CAPTION>` hints.
Owned resizing supplies 768×768 RGB; the processor's spatial transforms are
disabled and pointwise normalization retains its independent 1e-6 check.
FP16 eager inference uses one beam, a 256-token bound, TF32 off, seed31,
two host threads and a 0.35 CUDA allocator fraction. No paid API calls occur.

The twelve-witness stage generates twelve valid responses. The 384-source stage
reuses eight witness responses and produces 383 valid hints plus one capped,
repetitive response. Its nonzero exit is retained; no retry raises the cap or
hides that failure. There are 388 unique response contracts across both stages.
Captions remain review hints, never person/face/artistic labels. The source with
the invalid caption is separately inspected as a photographic market scene.

Producer 775bd2fc performs generation. Revision 844cb52a later corrects terminal
progress to expose the invalid hint; the prior progress file remains preserved.
The full stage takes 103 seconds under 4 GiB/two threads; run-heavy reports
peak RSS **1.88 GiB**, minimum available RAM 20,570 MiB and peak load 5.58.
CUDA peak allocated/reserved bytes are 648,503,296 / 713,031,680; these allocator
counters exclude some driver accounting. Full responses and tokens are cached.

All sixteen complete contact sheets and targeted full owned previews were
inspected by Codex. Raw observations and 23 explicit subsequent corrections
remain visible. Browser checks decode all384 images, validate all2,518 projected
boxes and report zero exceptions. Screenshots total43,463,511 bytes within the
registered96 MiB allowance. A separate replay of the shared template preserves
119 legacy images across12 sheets with zero exceptions; its screenshots total
10,730,673 bytes. Neither review supplies human artistic ratings.

Four sources are document/art content: a poster, painting, history article and
framed-photo exhibit. Seven further photographs have ambiguous person targets
involving sculpture, advertising, television, portraits or projection. They are
withheld from person auxiliary supervision pending per-box review. Photographs
with real museum visitors, shoppers and background people remain photographs;
a caption mentioning a statue does not automatically exclude them.

## Relationship barriers and candidate masks

The unchanged Hamming≤10 screen compares 170,688 pairs: 97,152 against253
available references and73,536 internal pairs. Six possible links affect seven
new sources, including two fitting/development pairs. All seven are quarantined.
There are zero exact encoded/working-RGB matches. Fingerprints do not establish
complete duplicate recall and these links are not claimed as confirmed duplicates.
No earlier heldout pixels are opened to resolve them.

The source mask is frozen before detector quality scores are read. Its SHA256 is
`e443d102d0069d89197c81e4c0032eff2335e4d9b85f01503a2248ed48f88f80`.

| Candidate purpose | Fitting | Development |
|---|---:|---:|
| Photographic composition sources | 280 | 93 |
| Person auxiliary supervision sources | 275 | 91 |

These are source eligibility masks, not labels or evidence of useful crop quality.
Person annotations may be incomplete; full person retention is not the policy
for every hero/detail request. Whole faces still require their separate guard.
Earlier source/component roles, crop arithmetic and face expectations are unchanged.
