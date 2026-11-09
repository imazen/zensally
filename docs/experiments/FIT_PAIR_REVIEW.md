# FIT-only nested-pair visual review

Prepare four photographic FIT sources without opening evaluation labels: two
with one weak head nomination, one with two, one with three or more. Within each
stratum order by SHA256 of`fit-pair-review-v1|source-id`, retain distinct source
families, and require existing source training eligibility and valid annotations.
Freeze identities before viewing crops. Nominations are not reviewed face counts.

For each source, request3:1 and1:3. Each option contains two matching-aspect,
integer-native rectangles: an outer Minimal crop of maximum possible area, and
a strictly smaller Maximal crop nested completely inside it. Existing candidate
banks and eligibility stay unchanged. Curate only exact-aspect rectangles with
no partial intersection of a cached weak head and at least one nominated head
retained. This is offline proposal curation using weak annotations, not a runtime
guard, complete face inventory or new quality label.

Offer at most three distinct pairs: highest summed selected-student score,
largest cached whole-pose/head-joint retention, and largest inner area. Break
ties deterministically with student score and candidate indices; avoid exact
duplicate native pairs. Keep current raw Minimal/Maximal choices as separate
diagnostics, explicitly recording whether they are nested. An empty curated
pool is a bank-availability result, not a contain or negative artistic label.

Use actual owned crop/resize/PNG pixels with independent exact RGB readback,
maximum output axis512 and128MiB storage budget. Preserve full-source views,
source URLs, exact coordinates, candidate indices and raw score identities.
No importance planes, new teacher inference, paid API calls or TEST access.

Record Codex visual comparisons as weak model judgments, never human ratings,
unanimous excellence or acceptable continuous intervals. A pair may have an
acceptable outer and poor inner, or vice versa. Do not admit every intermediate
crop, invent negative labels outside the viewed set, move source roles or change
existing label admissions. Any fitting protocol must explicitly name these new
weak judgments and preserve the original quality gate.
