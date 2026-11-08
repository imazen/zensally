# Pose-derived head constraints — 2026-11-08

Adding pose-derived regions repairs visible helmet cuts in the two-witness
screen, with substantial additional containment. This is a geometry result,
not an artistic model or a full-face-coverage claim. No learned weights,
shipping behavior, labels, canonical roles or paid API usage change.

The [registered rule](../docs/experiments/POSE_HEAD_BOUNDS.md) uses native
full-person ROI landmarks from the [101-photo screen](owned_photo_screen_2026-11-08.md).
Four poses yield four outward-rounded padded head rectangles; none is skipped
by the fixed confidence gate. All original face constraints remain, including
the possibly false ceiling-art detection. No heuristic removes a face box.

| Two sources × eleven ratios × two modes | Before | Added pose regions |
|---|---:|---:|
| Offered crops | 38 | 24 |
| Full-image containment | 6 | 20 |
| Offered crops clipping the four estimated regions | 27 | 0 |

Of 44 requests, 34 change: 14 switch from crop to containment and 20 move the
crop. Every new crop independently passes source bounds, exact integer aspect
and every supplied old/new region constraint. Protecting an estimate does not
establish that the estimate encloses every real head.

In 11048's 3:1 Minimal request, the crop moves from
`[0,1515,3000,2515]` to `[0,1262,3000,2262]`. The actual-pixel comparison shows
both foreground helmets retained after the move. Its 1:3 requests now contain
the source because the separated estimated heads cannot fit. In 11214, many
requests also contain: distant visitors and the retained ceiling-art face
constraint span much of the source. The two posed visitors are not every
visible person, so no complete coverage certification follows.

The optional additions leave default behavior unchanged: rerunning all 101
sources produces semantically identical JSON in all 103 original bank/summary/
witness files, covering 2,222 requests. New research-only arguments are paired,
bound to explicit source/bank hashes and limited to eight source IDs. The
existing 101-source cache, face padding, retention tolerance and scorer remain.

Resource records on i265: the default regression takes 6 s / 0.69 GiB peak RSS;
two-source proposals take 1 s / 0.68 GiB. Both use a 4 GiB run-heavy cap and
two-thread pool bounds. These are screen durations, not paired runtime latency.

The review renders eight actual crop PNGs across eight 3:1/1:3 tasks, with
original-source containment where required. All eight independently read back
to their owned RGB exactly. Browser checks decode 24 displayed images and
verify each cropped image's exact aspect, with no errors; run-heavy records
0 s / 0.19 GiB peak RSS. Independent readback records 0 s / 0.02 GiB. No SVG
crop projection or saved importance plane is used in this comparison.

The immediate evidence supports testing person-ROI pose features more broadly.
It does not justify promoting these hand-chosen padding constants, rejecting
unmatched face detections, training on this review-only cohort, or treating
more conservative containment as an artistic-quality improvement.
