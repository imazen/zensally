# Person-region face/pose suitability — 2026-11-08

This posthoc diagnostic tests two visually inspected, unadmitted TRAIN photos:
11048 (skiers) and 11214 (ceiling and visitors). Whole-frame sparse full-range
face detection missed visible people in both. This is not a representative
recall estimate, a new training target or an automatic acceptance rule.

Freeze the existing EfficientDet Lite0 person detections at threshold 0.3:
four in 11048 and five in 11214. Project their preview bounds to native pixels
with outward rounding. For each, evaluate two views: the full person box plus
10% context on each edge; and its upper 45%, plus the same horizontal context
and 10% of original person height on its vertical edges. Clip to source bounds.
Do not change detector thresholds or remove low-confidence boxes after seeing
results. Existing full-frame sparse face/pose outputs are the baseline.

Create all 18 views from frozen owned native RGB using the existing pinned
zenresize Lanczos worker. Preserve aspect ratio, cap the long edge at 768 and
never upscale above native size. Independently read back PNGs against these
owned RGB buffers. Use the already pinned MediaPipe sparse full-range face,
short-range face and Pose Landmarker Lite CPU references with their existing
thresholds. Keep coordinates, scores and visible landmark summaries, not maps.
Map detections back through exact integer crop/output dimensions; retain raw
coordinates and explicitly clipped visible boxes. No inference result becomes
an authoritative face annotation or artistic grade.

Bound this diagnostic to two source buffers, 18 view buffers, 54 model calls,
64 MiB of view artifacts and 4 GiB run-heavy memory with two CPU threads.
Measure process peak RSS, not tensor byte counts, and retain logs and failures.
Inspect results on both photos before proposing a broader experiment. No new
model download, paid call, source-role change or shipping behavior is involved.
