# zensally development notes

The shared crop types and model preprocessing live in `crates/zensally`.
General ONNX backends, the fixed w24 executor and embedded-depth experiment are
separate workspace members. See README.md and docs/DEPTH_IMPLEMENTATION.md for
current scope, numerical contracts and validation limits. No release of the
experimental crates is prepared.

## Known Bugs

### Fixed: partial channel blocks in w24 concatenation (2026-11-06)

The shared NN tensor concatenation requires full inner channel blocks. w24's
24-channel skip connections violate this requirement with 16-lane packing.
The experimental executor now handles partial blocks privately, preserving
logical channel order and the aligned shared fast path. The regression checks
8- and 16-lane layouts without requiring a particular CPU. Keep full-graph
scalar/SIMD coverage on actual AVX-512 hardware as well.

### Fixed: implicit std in default-free builds (2026-11-06)

The core's std feature previously did not activate a no_std configuration.
The new runtime also inherited std through its NN SIMD dependencies. The core
now uses alloc without std, retains dependency-free integer resize rounding,
and the owned NN patch provides private libm-backed float operations where
needed. CI checks aarch64-unknown-none; runtime tests also cover both std modes.

### Dependency review limitation

The consumer graph is verified with public pinned review commits. Several
standalone codec dev/optional graphs still conflict over SIMD beta.1/beta.2
requirements. Their numerical guards have not been relaxed. The review status
and reproduction details belong in docs/DEPTH_IMPLEMENTATION.md.


### Fixed: JPEG stored-coordinate agreement (2026-11-06)

The native JPEG decoder corrects EXIF orientation by default. Embedded geometry
is exposed in stored coordinates with an explicit orientation tag, so extraction
and the complete-image example now request stored primary/auxiliary pixels.
A non-square EXIF Rotate90 carrier verifies stored dimensions, display dimensions
and the cue's retained orientation. Decoder defaults are capped consistently and
returned float/confidence planes honor the output-byte limit.

### Fixed: outlined depthwise border SIMD (2026-11-07)

Direct padding reads initially introduced an outlined border closure. The
cycle profile exposed per-operation target-feature bridge calls. Keep the
border helper forced inline into the dispatched kernel; numerical operator
and complete-graph checks retain the original FMA order. The memory/CPU
measurement report records this intermediate regression and its correction.
