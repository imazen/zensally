# Crop-teacher previews

Offline tool using published zenresize 0.3.1. Its stable SIMD dependencies are
isolated from the experimental codecs' beta cohort. It adds no dependency to
the deployed detector. There is no codec here: the owned PNG example decodes
and encodes, while the worker receives strided RGB8 sRGB and emits raw previews.
A source crosses the process boundary once per bounded preview batch.

Build with `just crop-preview-build`. Set `ZENSALLY_PREVIEW_RESIZER` to the
resulting executable before running `crop_label_pairs` or the crop-pilot recipes.
With no `CARGO_TARGET_DIR` override it is
`tools/crop-preview/target/release/zensally-crop-preview`.
The parent uses 768-pixel source previews in the expanded study. Candidate
previews retain aspect within 262,144 pixels and a 1,536-pixel long-edge cap;
this keeps extreme ratios from collapsing to a tiny short edge.

`just crop-preview-check EXECUTABLE` checks packed/strided equivalence,
batched/individual output and invalid geometry/input lengths. Large builds
and corpus commands still need the host's normal resource wrapper.
