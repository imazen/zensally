# Offline JXL storage experiment

`zensally-crop-compress INPUT_RGB8 WIDTH HEIGHT NEW_JXL lossless|0.5`
encodes a packed, opaque RGB8 sRGB source at effort 7. It preserves dimensions
and refuses existing output paths. Use `tools/crop-png` to decode source PNGs
with our codec first. The separate processes isolate incompatible SIMD dependency
cohorts; no image decoder is added to the encoder's dependency graph.

Build with `just crop-storage-build`. Both tools retain locked dependency graphs.
`sunburst_recompress.py` checks the native PNG against a read-only independent
PNG decoder, then checks both owned and libjxl JXL readbacks. PNGs with explicit
color transforms are rejected pending a metadata-preservation policy. These
Sunburst sources are opaque RGB8, with no explicit color-profile chunks.

Original PNG bytes and embedded generation provenance stay in the native archive;
the JXL is a pixel-preserving derivative in lossless mode, not a byte-identical PNG
archive. Signed provenance is not transplanted onto a changed file format.
Distance 0.5 changes samples and is a separate comparison, not a lossless source.

The comparison runner accepts `--resume` while a source set is arriving. It
binds codec binary hashes and rechecks completed source/JXL/RGB hashes before
reusing them; a partial per-image attempt is preserved and refuses overwrite.
`sunburst_dataset.py` prepares regular files only after every planned source has
lossless readback evidence and a coordinator inspection. Git publication remains
a separate reviewed step.
