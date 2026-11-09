# Person-region interval geometry control

Registered before evaluation. Automatic region descriptors improve some crop
coordinates but still cut53–55/139 accepted DEV nominated heads. Test direct
geometric constraints from the same source-only cached Lite2 boxes, isolating
proposal geometry from what the weakly supervised ridge learns.

Use all72 frozen sources,11 ratios, frozen geometry and four person-region
ridge parents. Detector confidence0.3/max50 and exact source hash checks stay
fixed. Two policies use existing roi_geometry.person_views: full padded body,
and padded upper45-percent view. Both keep all detections without dropping
crowd/overlapping proposals; exact duplicate boxes alone may be deduplicated.
Upper view is a body-context heuristic, not an actual head detector.

Reuse refine_nested_origin.refine without changing its exact solver. Minimal
keeps maximum-area native aspect extent; choose the nearest origin containing
one complete proposed region and cutting none. Inner retains every proposed
region retained by outer, enlarging only as required and staying strictly
smaller. Other proposed people may be excluded entirely. Infeasible outer
means neither endpoint; no strict child means Minimal alone. No detections
means unchanged parent with an explicit unchecked status, never face safety.

Ten fixed controls; upper-view is the primary proposal hypothesis, full-body
is a conservative comparison, not a DEV-selected winner. Record all emitted
and missing endpoints, unchanged accepted-reference matching, independent
nominated-head cuts and retained heads, common-offer repairs/new cuts, explicit
teacher-contain emissions, interval safety, movement and enlargement. Report
FIT/DEV and essential/secondary separately; also report all nominated-head-known
requests, not only139 accepted DEV endpoints. Preserve each supplied-region
interval check, original labels/roles and all five quality evaluator files.

No new fitting, detector/teacher inference, API spending, TEST, runtime export
or promotion. Exact integer projection does not establish artistic composition.
Original frozen-bank quality is not rescored by this diagnostic; promising
geometry still needs independent original-quality compatibility and real pixels.
Run-heavy4GiB/two threads;16MiB total packet including at most two fresh source
witnesses, owned resize/encoding and independent PNG readback if warranted.
