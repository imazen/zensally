# Source-region feature control registration

Test whether explicit source regions address the direct crop scout's failures.
Keep its72 real families,48 FIT/24 DEV roles,525 accepted Sol weak-window
queries,471 explicit contain recommendations, C002 source features and exact
integer geometry. No API calls, TEST inference, synthetic inputs or production
changes. The original old-rank gate remains required and unmeasured here.

Five matched inputs append47 geometry values to the same2048 image features:
no regions, local sparse full-range faces, local Pose Lite padded heads, both
local cues, and Sol-nominated heads as an offline upper-bound input. Eight
area-sorted xyxy/mask slots are accompanied by extent, centroid and count over
all unique regions. Exact duplicates are removed. Omitted slots are recorded;
this descriptor does not represent complete face coverage or person identities.
Never supply crop windows, chosen-subject IDs or query decisions as input.

All five variants use identical-capacity32-unit direct heads, the original
source-balanced L1/offer-BCE objective and selection rule,108 epochs and
seeds31/137/911. Image, all-zero geometry-only and within-role/content-shape
permuted controls give45 fits. Permutation moves the image and its derived
region descriptor together. Normalize every column on FIT only. The zero-region
variant controls added capacity; do not compare it only to the smaller old head.
Repeated geometry controls must agree across variants for each seed.

Retain epoch-zero and selected native predictions, selected and terminal
optimizer/RNG states and every epoch's metrics. Hold the current best checkpoint
in bounded CPU memory and write it once per fit rather than writing each new
best to disk; earlier experiments remain unchanged. This is an artifact-storage
change, not a checkpoint-selection or scientific-gate change. Cap each variant's
evidence at128MiB and each process at4GiB/two threads.

Compare each region variant against the matched zero-region variant by paired
family bootstrap, including edge error, top1/best2 matching, raw nominated-head
cuts and offer Brier. Oracle improvement alone is not a runtime localization
result. DEV reuse, incomplete nominations and finite windows remain explicit.
Only a suitable candidate proceeds to rendered review, original-quality and
runtime qualification; no model is promoted by this feature scout alone.
