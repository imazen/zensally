# Embedded depth and fixed-graph saliency implementation

The requested scope includes JPEG and HEIC embedded-depth extraction, a reusable
geometric saliency experiment, current-main codec integration with patches in
PRs, recovery/search of earlier depth abstractions, and a custom w24 runtime
verified faster than tract on the same model and inputs.

## Source baseline

Resolved on 2026-11-06:
- zencodec main: `1fb997bb88e7405e6a315bf1f0b22f71e99be272`.
- zenjpeg main: `cae3a9f38786803a20575474792ead2a7d418f3b`.
- heic main: `4d72caddb4c1d664237a8d3693f35d69ac80caaf`.

Searched all fetched branches of these three repositories for shared decoded
DepthMap/DepthRepresentation types. Existing reusable types are JPEG's
`DepthMapData` (introduced by `b3a693c9`) and HEIC's `DepthMap` and representation
metadata. zencodec has `Supplements::depth_map`, but no decoded-depth payload on
the branches searched. Also searched zenpipe/retired zenimage source.

## Acceptance gates

- Decode real embedded JPEG and HEIC depth payloads with existing codec code;
  exercise GDepth/DDF/MPF, confidence, representation and alignment semantics.
- Validate strided depth views, missing/invalid-depth behavior, orientation and
  meaningful foreground/background examples. This remains an experimental cue;
  do not claim demonstrated crop-quality gains without measuring them.
- Reuse existing owned inference kernels before extending missing operations.
- Run the committed w24 weights, all operators and complete fixed graph; compare
  outputs against a separately implemented reference and tract. Report numerical
  disagreements explicitly rather than hiding them behind crop thresholds.
- Profile and benchmark custom and tract w24 under identical preprocessing,
  release settings, thread count and inputs. Record full commands and paired
  measurements. Faster-than-tract is unproven until those measurements pass.
- Submit public API and dependency changes as PRs assigned to lilith. Do not
  publish crates as part of this work.
