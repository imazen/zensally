# Frozen photographic cohort: pose head constraints

Extend the earlier two-image [pose head screen](POSE_HEAD_BOUNDS.md) as a new,
separate experiment. Reuse its unchanged `head_bounds` implementation; do not
change or replace the earlier protocol, inputs, confidence gate or results.

Use the existing72 FIT/DEV source records, cached normalized full-source
Pose Lite landmarks and native dimensions. Bind exact source inventory and
pose-result hash. Convert x/y to native pixels before calling `head_bounds`.
Record every returned bound and every confidence/geometry rejection; a missing
pose is unknown. No new teacher inference or paid API calls.

Run the existing fixed-bank candidate audit with padding25, both registered
whole-or-excluded policies, original and all selected replay controls. Add three
descriptive region controls: pose head bounds alone, tiled YuNet plus pose,
and all existing face outputs plus pose. Preserve every original face box.
Apply the same extra policy padding to all regions, including the already
conservative pose head estimates. No new tuning or winning-seed selection.

Head constraints permit a person to be selected while other heads are wholly
excluded. They do not require every person or whole body to fit. Candidate and
quality contracts stay fixed; no TEST, training admission or deployment change.

Compare the cached joint/body feature values for selected crops separately.
These reuse the registered composition features and confidence weights; they
are geometry diagnostics, not artistic grades. Check the retained group,
baseball and helmet witnesses with actual pixels and keep failures visible.

The [measured result](../../benchmarks/rehearsal_pose_constraints_2026-11-09.md)
retains a limited head-protection/containment tradeoff, lower accepted-reference
matching and inconsistent body diagnostics. Full neural inference reproduces
all417 tested guarded decisions; this does not justify promotion.
