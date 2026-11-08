# Broader owned-photo screening — 2026-11-08

Acquire the remaining PD-own photographic screening candidates from the pinned
current imazen-26 JXL-sRGB TRAIN export. This is input preparation and local
model screening, without crop judgments, fit admission or a new split.

The existing audit-v2 metadata yields 101 sources / 98 families after excluding
all source IDs, families and exact original/pixel/encoded hashes in the existing
fit/dev/eval/test selection. Categories: 21 general, 11 interiors, 54 nature,
15 food. No people-category image is added by this owned subset; category tags
are screening hints and incidental people remain possible. Unsplash acquisition
stays separate. No new third-party rights admission is inferred.

Reconcile each entry against the pinned export/family/historical manifests,
using the existing source-ID rule and all manifest hashes. Preserve original
URLs, licences, provenance and family IDs. Canonical TRAIN remains the source
bucket; `screening` denotes an unadmitted use, not a new train/test assignment.
No previously fixed role moves. All records stay training-ineligible,
human-unrated and without certified face coverage.

Use the existing owned decoder, PNG encoder and zenresize reference executables.
Verify each JXL hash/declared byte count and decoded dimensions. Produce an
aspect-preserving 256-square letterbox tensor and a 768-long-side preview;
padding lies outside valid_rect. Never use an independent decoder to generate
model inputs. Independently read back the resulting source/preview PNGs as
test oracles, comparing them with owned RGB. Save coordinates/scalar screening
summaries; do not save metric diagnostic maps or invent artistic labels.

Declared encoded payload is 684,508,794 bytes. The sum of RGB8 dimensions is
3,588,426,810 bytes for disk planning, not a measured memory footprint. Reserve
12 GiB for acquisition outputs and require at least 16 GiB free before starting.
Use one acquisition worker under a 6 GiB run-heavy host cap and two-thread pool
limits; record actual peak RSS and disk usage afterward. Keep full progress/logs
and all partial results. Mirror completed inputs and generated evidence to NAS
and verify complete membership hashes independently.

Broader local face/pose/crop proposals may support a review gallery. Those
outputs are imperfect geometry/model suggestions, without certified face
coverage or training eligibility. No paid API calls, learned-weight updates,
canonical holdout image reads or model promotion belong to this screen.

The runtime extension uses the independently checked 256-square owned tensors
and 768-long-side previews, with source hashes binding both to native pixels.
It caches no targets. Local sparse full-range face and pose outputs are compact
coordinate records. Candidate search protects the supplied detector regions
without claiming complete face coverage. Keep all proposals undeployable.

Extend the unchanged crop-query numerical contract to this cohort using a
bounded neural-input fixture: 101 fixed 32×32×17 feature arrays, scalar query
scores, candidate boxes/masks and six normalized image witnesses. These are
required runtime inputs, not metric diagnostic maps; reserve 40 MiB for that
fixture. Select the six witnesses deterministically from source metadata across
the four categories before looking at numerical errors. Never save importance
planes. Existing six-AI-witness defaults and earlier fixtures remain unchanged.
