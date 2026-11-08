# zensally development notes

The shared crop types and model preprocessing live in `crates/zensally`.
General ONNX backends, the fixed w24 executor and embedded-depth experiment are
separate workspace members. See README.md and docs/DEPTH_IMPLEMENTATION.md for
current scope, numerical contracts and validation limits. No release of the
experimental crates is prepared.

## Known Bugs

### Fixed: direct API contain-zone parsing (2026-11-08)

The first joint ROI/crop response expressed containment with a full-source
percentage box. Rejecting all nonempty contain options was too narrow for this
new protocol. Full-source options now require exact 0–100% bounds, and the saved
completed response is recovered without another POST. Completed raw-response
caching is separate from annotation validation/admission.

### Training-tool operational observations (2026-11-08)

The external benchmark runner canonicalizes a virtualenv Python symlink and
thereby loses its site-packages. Invoke `/usr/bin/env` with the virtualenv path
as an argument. Preserve the original failed packet; it supplies no timing data.
The installed profiler used system scratch despite TMPDIR; use explicit
NSYS_TMPDIR in profiling recipes. Final Nsight reports remain retained outside
scratch. CUDA allocator counters exclude some driver/private-pool accounting;
always retain reserved bytes, device telemetry and host RSS separately.

### Fixed: first CUDA graph loss snapshot (2026-11-08)

CUDA graph capture records kernels without executing them. Reading its loss
before an explicit replay observes an uninitialized value. The training probe
now replays/synchronizes before its initial snapshot and drops the side-stream
warmup loss before capture. The first graph attempt is retained as diagnostic
evidence; its initial loss is invalid. Training memory reports retain both
allocated and reserved bytes; graph-private pools prevent interpreting a low
live-allocation counter as total device memory.

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

### Fixed: shared Maximal crop integer bounds (2026-11-07)

The aspect-ratio quality harness rejects Maximal rectangles extending beyond
the source. The initial full evaluation stopped in all four shards; partial
tables are not quality evidence. zenpipe 0007edcf clamps the integer origin
while preserving the rounded crop extent; synthetic horizontal/vertical
regressions and std/alloc-only checks pass. The consumer pins that correction
and retains strict crop validity for the complete rerun.

### Fixed: heatmap crop placement can discard the dominant region (2026-11-07)

The user's gallery case ILSVRC2012_test_00084919 / canvas-1x8-p0 / stretch
has a strong upper heatmap band and weak lower speckles. The 9:16 Minimal
rectangle starts at y=637 and retains zero labelled foreground: the shared
solver's bounding-box adjustment moves a crop off the dominant band. The
Maximal rectangle also misses it. zenpipe 6ed32dea now maximizes retained
squared map mass at the existing crop size. Both 9:16 modes start at y=60 and retain 100% of the labelled subject.
The complete eight-arm rerun retains every old mask metric and crop size;
see benchmarks/w24_crop_placement_blur_2026-11-07.md. Original artifacts are
preserved; the original report's crop interpretation is superseded.


### Fixed: crop-teacher test scratch setup (2026-11-07)

The request-builder fixture assumed the user's scratch parent already existed.
Fresh CI users lack that directory. The test now creates its scratch parent
before allocating its temporary fixture; all role, mapping and admission
assertions remain intact. Reproduction uses a mocked fresh home path.


### Fixed: crop pilot offered only Minimal endpoints (2026-11-07)

The first annotation renderer offered the leftmost/rightmost maximum-area
Minimal crops. A forced preference between those cannot establish good framing.
The revised renderer consumes a center/interior/vision proposal pool and
independently validates source bounds, exact aspect and Minimal area. Terra
inspects rendered proposals and may reject all. Historical labels stay unadmitted.


### Fixed: overnight CLI annotation evidence capture (2026-11-08)

A midnight environment update can appear after the annotation prompt in the
same CLI turn. Taking the last user message then rejects a valid completed
judgment. Capture now requires exactly one matching annotation prompt inside
that completed turn's boundary. Missing/ambiguous prompts still fail; all
candidate-ID, image-hash and admission checks remain unchanged. Synthetic
midnight, stale-turn and ambiguous-prompt fixtures cover the boundary.
