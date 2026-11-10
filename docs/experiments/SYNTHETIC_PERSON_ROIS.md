# Synthetic person-ROI teacher suitability

Registered 2026-11-08 after the two real-photo pose-head witnesses. Broaden the
screen before deciding whether to fit a small head-localization student.
This phase admits no artistic labels and does not train on the new real photos.

Reuse the existing 56 synthetic source registry with its unchanged 48-fit /
eight-development source and brand/scenario family roles. Join by exact owned
native-RGB hashes to the pinned runtime manifest and EfficientDet Lite0 results.
At the existing 0.3 threshold there are 117 person boxes: 94 on 36 fit sources
and 23 on seven development sources. Keep all boxes, including uncertain or
false person detections. No missing detector result is a person-free label.

Render one full-person ROI per box with the previous exact 10% context rule,
native pixels, owned zenresize Lanczos, aspect preservation and no upscaling,
long edge at most 768. Upper-body windows remain in the earlier negative probe;
they are not repeated. Independently read back every owned PNG against RGB.
Bound to 117 views, 512 MiB of view artifacts, 4 GiB host memory / two threads,
and require 8 GiB free disk before starting. Preserve all failures and raw
coordinates. Do not save importance planes or segmentation maps.

Use only the already pinned sparse full-range face detector and Pose Lite CPU
references, with unchanged thresholds: 234 calls total. Map all detections and
landmarks to native source coordinates, retaining original ROI coordinates too.
The fixed pose-head rule may form weak head targets; a target outside its input
ROI is marked unavailable for ROI-local fitting, not silently truncated into
ground truth. Missing poses/confidences remain unknown. Positive targets are
still noisy estimates, not face certification or a head-count annotation.

Before any fit, report teacher coverage by role, source and family, unmatched
faces/poses, off-ROI targets and representative disagreements. A student-fit
follow-up requires usable pose targets from at least 12 fit families and three
development families and a separately registered fit/control protocol. No role
may move to meet those counts. Keep all sources within their original roles;
count multiple ROIs from one source/family together in later uncertainty work.
No paid model call, canonical holdout image or new model download is involved.
