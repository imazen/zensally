# Offline photographic subject references — 2026-11-09

Local phrase grounding can supply noisy known-region supervision, but cannot
certify complete products or people. Visually plausible product boxes coexist
with prop substitutions, partial subjects and missed people. No new reference
is admitted as artistic gold or complete-face coverage.

The registered87 queries cover72 fitting and15 development sources in the fixed
photographic registry:50 products,20 people scenes,11 prepared-food scenes and
six animals. All calls terminate without parser/truncation errors and produce
160 valid native-coordinate boxes. The other32 fit/development sources have
unknown subject references, not negative labels. No TEST source is opened.

All87 source/box overlays are browser-checked and visually inspected at bounded
preview resolution. Two clear semantic failures matter for supervision:
`4kbIkkch_Z8`'s seated-reader query boxes the phone; `kRnM-UJpK6I`'s counter-reader
query boxes the book/counter. A separate uniform `The people.` control therefore
runs on all20 people scenes,18 fit/two development, preserving the original
queries and outputs. It produces21 valid boxes; substituting these gives143
boxes across the87 sources. The phone and counter examples now box the people.
However, the generic beach-cafe result covers fewer patrons than the original
query, and the candle portrait can expand to nearly the entire source. Neither
phrasing is exhaustive. A compound smartphone/bottles query still returns only
the phone; some animal boxes cover only the visible head/body portion. Multiple
entities, clipped source subjects and tiny faces need separate uncertainty.

The fixed bank and independent integer feasibility calculation separate three
issues, without generating any new teacher scores or changing candidate masks:

| Known-box request status /1,914 | Original phrases | Generic people substitution |
|---|---:|---:|
| Bank contains all returned subject boxes |705|734|
| Requested amount feasible, sampled bank misses it |94|95|
| Only a least-crop fallback is feasible |69|67|
| No exact-ratio crop can contain all returned boxes |1,046|1,018|

These counts cover11 ratios and both crop amounts. They refer only to the
returned boxes, not complete scene subjects. The source-based set contains
multiple related photographer images and is not family-balanced in this table.

On329 known, informative development requests, the existing purpose-unspecified
student contains all original query boxes117 times, and its FocalNet teacher121
times. With generic people substituted, those counts are146 and152. One of the
330 requests has no informative original ranking target. These are unchanged
crops evaluated against different weak references, **not a model improvement**,
an artistic grade or proof that the other crops are wrong for hero/detail use.

Both87-source viewers decode all images and verify160/143 overlay boxes with
zero exceptions. Their18 screenshots total15,190,351 bytes, below24MiB. The
first generic-viewer launch fails because the chosen scratch path makes a
Chromium Unix socket too long; a short scratch path succeeds. Both logs remain
retained; no coordinate or browser expectation changes.

run-heavy on i265 reports initial grounding18s, peakRSS1.94GiB,
min-available21,831MiB, peak-load4.02; generic grounding6s/1.99GiB/23,676MiB/
2.08. Both use6GiB/two-thread host caps. CUDA allocated/reserved peaks are
648,519,680/700,448,768 and648,512,000/700,448,768 bytes. Initial/generic
geometry audits take2s/1s, peakRSS0.61GiB, minimum available23,767/25,041MiB,
peak-load3.17/2.00. Browser checks take2s/1s, peakRSS0.20/0.21GiB,
minimum available24,690/25,023MiB, peak-load2.91/1.73. These are resource
observations, not controlled runtime comparisons or GPU-saturation claims.

Retain these references for a separately registered candidate-local retention
experiment with image/geometry/permuted controls. Only fitting-source known
boxes may supply noisy gradients; a learned retention score cannot certify
unobserved subjects or replace the face guard. Catalogue whole-subject retention
and hero/detail composition remain distinct. No model is promoted, no old rank
weights or quality gates change, no diagnostic maps or paid API calls occur.

See the [protocol](../docs/experiments/PHOTO_SUBJECT_RETENTION.md) and
[artifact pointer](photo_subject_retention_2026-11-09.pointer.md).
