# Aspect ratio and overlapping-refinement experiment

Frozen before inspecting method quality, 2026-11-07. This is an evaluation
harness; model weights, training membership, existing APIs and numerical
regression gates are unchanged. Segmentation-mask retention is a crop-quality
proxy, not a human aesthetic preference score.

Data: all 5,019 DUTS-TE RGB/mask pairs from the existing local archive
SHA256 d9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104.
[DUTS](https://saliencydetection.net/duts/) supplies human foreground masks.
Do not redistribute these images in git. Image/mask hashes are frozen in a
manifest before inference. SHA256(id) first byte modulo four == 0 selects
calibration, others select evaluation. Model training used DUTS-TR; historical
model selection has already inspected DUTS-TE, so this is not a pristine
architecture holdout. Method comparison preserves the new partition.

Compare fixed rules: square stretch; black letterbox; mean-color letterbox;
four independently stretched quadrants; global stretch plus up to four
uncertainty-selected overlapping square views; and global stretch plus
aspect-aware covering views. No mask or quality target enters view selection
or inference. Evaluate each fixed method without tuning fusion on evaluation.

Each local map is projected into source coordinates. Taper overlap weights
at view edges; retain the global map where no view covers a pixel. Fuse with
alpha=0.5*global_probability, limiting promotion of globally unsupported
local foreground. Select normal-aspect views by global foreground plus
uncertainty, penalizing overlap with already selected views. Cover extreme
aspect ratios with full-short-axis square views at >=50% overlap. Bound the
covering ensemble at sixteen views by enlarging windows when necessary;
letterbox enlarged rectangular views rather than stretch them.

Natural images are reported separately from synthetic stress canvases.
Stress uses deterministically selected 64 calibration and 64 evaluation
sources, extended to 1:1, 2:1, 4:1, 8:1 and reciprocal aspects, with content
at left/center/right (or top/center/bottom). Preserve source proportions and
foreground labels; report synthetic background construction explicitly.
Cases from a source always inherit its partition. These test geometric
sensitivity, not natural-panorama performance.

Use a common 256x256 normalized scoring grid for saliency MAE and thresholded
IoU/F-beta (beta squared 0.3, fixed probability 0.5). For source coordinates,
constant-axis scaling preserves relative area; grid discretization remains
an acknowledged limit. Test shared-solver Minimal/Maximal crops at 1:1,
16:9, 9:16, 4:1 and 1:4. Report foreground retention, feasible-mask-oracle
retention and regret, including failures/empty predictions. Do not use a
prior runtime's rectangle tolerance as a quality criterion.

Generate a browsable gallery of source/mask, each heatmap, actual extracted
crops and exact crop coordinates. Select examples deterministically and by
largest method improvements/regressions, label which, and retain complete
per-case tables. Include original URLs for cited papers and dataset sources.

Reader correction before the complete rerun: some frozen masks are RGB with
small channel differences. Match the training loader's grayscale convention
(BT.601 fixed-point luma, then threshold at 128), and independently compare
every decoded native binary-mask hash against Pillow convert("L"). Do not
change mask bytes or omit these sources. The strict-gray reader's partial
run is retained as failed evidence, separate from quality results.
