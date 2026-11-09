# zensally development notes

The shared crop types and model preprocessing live in `crates/zensally`.
General ONNX backends, the fixed w24 executor and embedded-depth experiment are
separate workspace members. See README.md and docs/DEPTH_IMPLEMENTATION.md for
current scope, numerical contracts and validation limits. No release of the
experimental crates is prepared.

Synthetic source generation follows [art direction](docs/experiments/SYNTHETIC_ART_DIRECTION.md):
review believable scenes and person/product/copy associations before generation,
then inspect full-size physical geometry before positive admission. Source review
overrides bind exact IDs and hashes; retain rejected originals and preserve splits.

## Known Bugs

- Source-aware PhotoFramer pair comparison chooses B in both orders on five
  of six calibration pairs. Independent single-image processing replays all36
  blocks and verifies real crop swaps; only1/6 swaps is consistent. The local
  critic is not admitted as a labeler. See benchmarks/source_aware_critic_2026-11-09.md.

- Resolved2026-11-09: unquantized critic12GiB GPU/8GiB CPU placement fails
  on a transient transfer after four responses. Separate10/10 placement finishes
  all16 with identical shared prefix outputs. It still fails quality qualification
  (four correct pair orders, one tie, one reversal); no label admission. See
  benchmarks/critic_precision_2026-11-09.md.

- Automatic full-person pair controls still cut55/139 accepted DEV nominated
  heads. Largest-person cues recover a playground child but introduce a partial
  helmet/head in a tall work-site crop. No promotion; see
  benchmarks/person_region_pair_control_2026-11-09.md.

- Low-capacity nested ridge controls still cut53–57/139 accepted essential DEV
  heads and emit85 teacher-contain requests. FIT-family-selected image/attention
  edge-error differences have bootstrap intervals crossing zero. Resolved:
  preflight now retains the secondary-only FIT family as a fold without using
  it for primary lambda selection; all99 targets/29 folds stay fixed. See
  benchmarks/nested_ridge_control_2026-11-09.md.

- The continuous nested-pair head preserves native geometry but image and
  aligned/permuted/measured attention controls all have worse paired DEV error
  than geometry-only. Selected heads leave56–66/139 nominated-head cuts and
  always emit85 teacher-contain essential DEV requests.23760 cached-descriptor
  CPU/GPU rectangles replay exactly; no runtime export. See
  benchmarks/continuous_nested_head_2026-11-09.md.

- Exact nested origin refinement protects supplied heads across each interval
  but still clips a missed left-hand head in the3:1 counter scene and makes
  classroom3:1 unavailable. Primary seed137 repairs15 and introduces5 cuts on
  common offers, with23/132 cuts and seven lost accepted endpoints. Original
  frozen-anchor composition regret slightly worsens. No promotion; see
  benchmarks/nested_origin_refinement_2026-11-09.md.

- Nominated-head oracle refinement reaches zero cuts on132 offered accepted
  endpoints but only22–27/139 composition matches; automatic proposals leave
  22–23 cuts at the same offer count. Cached nominations are incomplete weak
  labels and oracle membership differs; no face/artistic certification. See
  benchmarks/nested_oracle_diagnostic_2026-11-09.md.

- Person-conditional soft risk preserves full original/new offer coverage and
  all three attention seeds pass the original gate, but the strongest run still
  cuts40/139 newer accepted endpoints, repairing11 and introducing6. Classroom
  and1:3 waterfall face failures replay through the full network exactly. No
  promotion; see benchmarks/person_risk_routing_2026-11-09.md.

- Candidate cut classifiers do not supply a safe crop policy: aligned attention
  heads leave20–21 cuts while offering98–103/139 newer accepted endpoints and
  retain only189–204/448 original endpoints. All15 guards lose full original
  coverage. Cached compact/CUDA routing is exact; visible classroom/waterfall
  failures remain. See benchmarks/candidate_cut_classifier_2026-11-09.md.

- Resolved2026-11-09: original attention preparation initially assumed a
  dictionary manifest, while the pinned attention packet has size-bound list
  records. It failed before outputs; the corrected schema reader and original
  failure log are retained, with all scientific inputs unchanged.

- Newer accepted-reference head diagnostics cover139 essential DEV endpoints;
  the raw ranker also emits candidates on contain/unknown requests. Expanded
  nominated-head diagnostics find166 cuts among396 essential requests on22 DEV
  sources. Frozen bank feasibility is also limited: only161/198 ratio pairs can
  retain a nominated head in a strict safe pair. No universal-face or full-output
  safety claim follows from the narrower score. See benchmarks/face_label_coverage_2026-11-09.md.

- Fixed-endpoint pair correction passes the original raw/nested quality gate
  for six of nine fits but retains visible face cuts in the counter group and
  adds a weak head cut in the child scene. Additive and permuted controls each
  pass as often as the real bilinear head; meaningful relationship benefit is
  unproven. Full neural choices replay exactly. No promotion; see
  benchmarks/pair_relationship_training_2026-11-09.md.

- Resolved2026-11-09: the pair-correction renderer requested a real-seed137
  head repair although none exists. The first attempt failed before output;
  retained evidence now uses the measured additive-seed31 repair. Existing
  annotations, quality gates and crop geometry remain unchanged.

- Structured pair fitting remains unqualified: all nine matched replay,
  endpoint-top and pair-top controls fail the unchanged raw-plus-paired
  original quality gates. FIT-only capacity training increases exact pair
  agreement196/1024 to433/1024 while held-out paired regret worsens.
  Numerical replay has no tested crop changes; no runtime model is promoted.
  See benchmarks/structured_pair_training_2026-11-09.md.

- Strict nested pair selection does not certify quality: outer-first clips
  the head on forest sourceaf8912bcf7fae386 at1:3 Maximal, and the source bank
  lacks a strict child for31/216 newer DEV essential raw Minimal origins.
  Full neural inference reproduces all8404 tested pair decisions, so these are
  retained model/selection failures. See benchmarks/nested_selection_2026-11-09.md;
  the original quality threshold is unchanged and every structured policy fails it.

- The research ranker's independently selected Minimal/Maximal outputs are
  not structurally nested: five of eight requests in the four-FIT-family pair
  review violate nesting. Curated teacher options enforce exact matching-aspect
  nesting, but that does not repair model selection. See benchmarks/fit_visual_preferences_2026-11-09.md.

- Unconditional union face constraints visibly regress nonhuman crops in the
  original DEV panel: a watch becomes almost blank, honey label content is
  excluded, and mushroom caps are clipped because of false face proposals.
  Conditional weak regret also regresses and all guarded policies lose full-panel
  coverage. See benchmarks/original_guard_compatibility_2026-11-09.md;
  no source exceptions, gate relaxation or promotion follows.

- Resolved2026-11-09: the first original-panel guard audit attempted geometry
  on zero placeholders in ineligible candidate slots. The caller now evaluates
  only existing eligible rectangles; validation and candidate masks stay fixed.
  The failed attempt is retained separately from successful quality results.

- Local face candidate filtering does not close the selected student's safety
  failures: tiled YuNet with padding10 and one retained detection leaves30/135
  offered seed31 crops cutting separately nominated heads, and the known counter
  group cut remains pixel-identical. The baseball crop clears a weak head flag
  while retaining body truncation. See benchmarks/rehearsal_face_admissibility_2026-11-09.md;
  no promotion or complete-face claim follows.

- The selected contrastive-rehearsal seed31 remains unsafe as a raw ranker:
  source9b15ccd240edae57 at1:1 Maximal clips a head at the edge;
  ff5b19334d988f53 at3:1 Minimal truncates the foreground body and leaves
  background fragments. The original-quality gate passes but does not certify
  face/artistic safety. Full neural replay reproduces the displayed decisions;
  retain the failures in benchmarks/rehearsal_witnesses_2026-11-09.md.

- Resolved2026-11-09: the first native residual exporter serialized Torch
  output-major dense weights to an input-major baker, failing1000 crop gates
  before timing. The rectangular-layer regression fails before the transpose
  correction. Original failed model/results remain preserved; no timing or
  quality threshold is relaxed.

- Resolved2026-11-09: the bounded relationship viewer initially inherited a
 192-pixel ratio unit that requested3072x1728 for16:9, exceeding its pinned
 resizer's bound. The new caller bounds its ratio unit to a768-pixel maximum
 axis; existing caller defaults and crop coordinates remain unchanged. Original
 failed output is retained. The baseball witness still has poor body framing
 despite a favorable nominated-head label; no artistic-quality admission follows.

- Resolved2026-11-09: the first cached scalar fit compared CPU output bits
  against CUDA-cached scores and stopped before training. Every arm now must
  replay the common CPU initializer exactly; the separate CUDA comparison
  records max absolute difference4.76837158203125e-7 and zero eligible-choice
  changes. Original failure is retained; serving numerical gates are unchanged.

- Resolved2026-11-09: the first nested-range probe allowed a loose excellent
  outer frame smaller than the actual Minimal crop. Protocol2 requires the
  maximum-area outer at the matching aspect; it rejects rather than enlarges
  a teacher's smaller frame. The original paid response remains calibration
  evidence and supplies no Minimal/Maximal interval label.

- Resolved2026-11-09: real-attention caching assumed fixed donor indices were
  tensors; the retained parent stores a list. The first attempt failed before
  feature writing and remains archived. Integer conversion preserves the donor
  assignments; cached C002 scalar replay is exact. Attention progress now marks
  terminal PID null; original as-run records retain separate PID verification.

- Resolved2026-11-09: the first exact-origin gallery displayed full-source panes
  for invalid annotations/unrequested ratios without distinguishing those cases
  from geometric containment. The viewer now says no audit was performed; old
  output is retained and coordinates are unchanged. Automatic origin constraints
  still visibly truncate the3:1 singer and remain unsuitable as a face guard.

- Resolved2026-11-09: the first direct-crop scout reused an exclusive artifact
  writer for mutable progress and stopped at epoch zero before any optimizer
  step. The failed state/log is retained; the progress writer is corrected
  without changing the registered data, loss or selection rule. The earlier
  bank-transfer audit also now converts NumPy integer scalars before calling
  strict native geometry validation; box coordinates and validator stay fixed.

- Resolved2026-11-09: a response-view helper wrote completion before its final
  provenance hash and passed Path to a bytes-only SHA helper. It now validates
  existing task/response bytes and writes provenance before completion. The
  original partial marker remains retained; labels and crop coordinates did
  not change, and no API request was repeated.

- Experimental subject-choice frozen-head scout remains unsafe as a raw crop
  scorer: source424084990ff2e8d7 at3:1 Maximal adds a face cut;
  9fa07d98319d63b9 still cuts foreground faces in square/tall crops.
  These diagnostics bypass shipping guards and cannot justify promotion.
  The later broad-Maximal/hybrid banks improve weak reference matching but
  retain these visible face failures. See benchmarks/subject_choice_fitting_2026-11-09.md
  and benchmarks/subject_bank_controls_2026-11-09.md.

- Resolved2026-11-09: the new subject-choice expansion initially included a
  960x480 request below the provider pixel floor. Preflight rejected the entire
  batch before any POST. The plan now uses1152x576 and validates every job
  through the same request validator before writing its manifest.

### Fixed: source rotation metadata uses decimal notation (2026-11-09)

The first MIAP image-information join accepts `90` but rejects the official
`90.0` notation, marking every joined rotation unknown. Exact finite quarter
turns now normalize to integer degrees; missing, fractional and nonfinite
values remain unknown. The original join and raw fields remain retained.
No image is acquired or admitted by this metadata correction.

### Fixed: SVG crop previews exposed surrounding pixels (2026-11-08)

An SVG viewBox selected crop coordinates without clipping the source image.
Aspect-ratio gutters could therefore expose pixels outside the selected crop.
Both research SVG builders now clip the image explicitly in source coordinates.
Four screenshot gutter probes fail on the retained old viewer and pass after
the correction. Stored crop coordinates and rendered-PNG training labels are
unchanged. Old SVG screenshots and ratings are unreliable visual evidence;
new localStorage namespaces preserve them without admitting them as labels.

### Fixed: inactive experimental heads advanced AdamW state (2026-11-08)

The first joint geometry fitter computed concatenated crop/auxiliary outputs
during both tasks. Zero gradients still advanced inactive-head momentum and
decay. Task-specific forward paths now leave inactive gradients absent; a
regression warms AdamW state and requires unchanged inactive weights in both
directions. Original joint runs remain retained as confounded observations.
All nine task-isolated repeats still select epoch zero on crop development
loss. Source roles, numerical/face guards and earlier artifacts are unchanged.

### Fixed: MediaPipe front stride-16 anchor order (2026-11-08)

The converted front model flattens six stride-16 anchors per NHWC grid cell.
The decoder previously iterated two anchors over the full grid three times,
misplacing stride-16 predictions. Consecutive equal-stride layers now merge
before visiting cells. A regression fails on the old ordering and passes with
six centers per cell; the stride-8 order and anchor count remain unchanged.
The runtime-model study retains pre-fix outputs separately from corrected ones.


### Crop pilot uses older corpus split assignments (2026-11-08)

The PNG-v3 pilot split is not interchangeable with current JXL-sRGB assignments.
An exact original-file SHA256 join places pilot TRAIN 6080 in current VALIDATE
16080. Nine of 32 pilot sources have exact original-hash matches; the other 23
need rendition/lineage reconciliation. These historical pilot labels remain
training-ineligible. The separate current-JXL fitting study now uses a pinned
family manifest that excludes historical holdout exposure; it does not transfer
the pilot labels. See DATA_PROVENANCE.md and
benchmarks/crop_fitting_2026-11-08.md. Keep the frozen pilot unchanged.

### Fixed: i686 depth corpus acquisition (2026-11-08)

The cross container's corpus downloader failed before the HEIC assertions ran.
CI and `just depth-i686` now run the same pinned corpus test on the host first,
then reuse its shared writable cache inside cross. The native regression passed
locally; the i686 test retains every encoded/depth hash assertion.

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
The root-run installed profiler used system scratch despite both TMPDIR and
NSYS_TMPDIR. A subsequent user-run trace honored the configured scratch path;
its GPU hardware metrics were disabled. Do not treat the overrides alone as
proof: verify the actual reported path. Final Nsight reports remain retained
outside scratch. CUDA allocator counters exclude some driver/private-pool accounting;
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
