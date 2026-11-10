# Offline PNG reader

`zensally-crop-png INPUT_PNG NEW_RGB8` decodes an opaque RGB8 PNG through zenpng,
retaining row order without resampling. Dimensions are emitted as JSON. This
study adapter rejects other channel layouts and keeps a bounded decode envelope.
It runs separately from `tools/crop-compress` because their pinned codec versions
use different SIMD dependency cohorts.
