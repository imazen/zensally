# Pose-derived head bounds: two-witness geometry screen

Registered 2026-11-08 after the native person-ROI probe. This is a deliberately
conservative geometry hypothesis, not a learned model or a face guarantee.
Retain all original face regions, including possible false positives. Do not
remove a face box because it disagrees with a pose.

Reuse only the full-person ROI Pose Lite results for 11048 and 11214. Upper
views remain a negative comparison in the earlier frozen packet. A pose has
usable head geometry only when all facial points 0–10 and shoulders 11–12 have
finite x/y and presence/visibility at least 0.5. Missing geometry is unknown,
not a negative label. Keep skipped reasons and all raw pose identities.

In native pixels, take the extrema of points 0–10. Set width to the larger of
2.4 times their horizontal span and 0.65 times shoulder Euclidean distance;
height is 1.3 times width. Center x on the facial extrema. Place bottom at
maximum facial y plus 0.25 times width, then top = bottom minus height.
Round bounds outward and clip to the native image. These deliberately padded
boxes can preserve substantial context and may still miss helmets or unusual
poses. The constants are fixed before inspecting resulting crop selections;
do not tune them on these two witnesses.

Add these regions to the existing detector head bounds and independently audit
the original 44 proposals. Then use the unchanged exact integer candidate
generator, original frozen FocalNet scorer, retention tolerance 0.05 and crop
modes. Compare changed boxes/actions and containment; retain all original banks.
No inferred region is a human label, no crop becomes deployable, and no source
role, training target, metric threshold or numerical contract changes.

Bound to two images, 44 requests and eight ROI pose instances, using the
previous frozen owned runtime cache and split model. Save coordinates/scalar
results only, no importance planes. Run under 4 GiB / two-thread host bounds.
