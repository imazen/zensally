# Synthetic person-ROI suitability — 2026-11-08

The local teacher produces enough distinct-family head targets for a bounded
student experiment, with important missing coverage. This is not a face-count,
artistic-quality or full-head-protection benchmark.

The [registered screen](../docs/experiments/SYNTHETIC_PERSON_ROIS.md) preserves
the existing 56 synthetic source roles and all 117 EfficientDet person boxes:
94 on 36 fit sources and 23 on seven development sources. All 117 owned ROI
PNGs read back independently to exactly matching RGB. Actual view payload is
132,885,294 bytes. Preparation takes 1 s / 0.03 GiB peak RSS; independent
readback takes 0 s / 0.03 GiB under run-heavy on i265.

The 234 CPU reference calls take 3 s / 0.23 GiB peak RSS. They produce 102 raw
pose instances and 125 face instances. Twenty ROIs have no pose. Thirty-two
pose-derived head estimates extend outside the input ROI and remain unavailable
for ROI-local fitting; none is silently truncated into a target. There are 65
usable weak targets: 53 on 31 fit sources / 21 families, and 12 on seven
development sources / seven families. This exceeds the predeclared minimum
support of twelve fit / three development families, without moving any role.

Sixty face boxes do not intersect an available pose-head estimate. Missing or
off-ROI poses contribute to that count; it is not sixty false faces. No missing
pose/detector result becomes a negative person/face label. Multiple detections
and targets within a source remain correlated.

Four deterministic hash-selected visual witnesses cover fit/development targets,
off-ROI targets and face-without-pose cases. The beach close-up has a large head
target touching source boundaries. The park scene's target describes the
standing person while another head is visible in the foreground. The indoor
off-ROI scene includes a partial foreground head. The seated indoor person has
a clear large face but no pose. These witnesses reinforce that a usable target
can describe one person without certifying everyone in the ROI.

No learned update or artistic label is part of this screen. A separately
registered fit may admit these 53 fit-only noisy geometry targets under their
existing source-family roles; development targets remain selection diagnostics.
The 101 additional real photos remain outside fitting. Model thresholds,
source roles, canonical holdouts and API spending are unchanged.
