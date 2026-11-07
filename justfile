# zensally development recipes

# Run tests (default features)
test:
    cargo test --workspace

# Run tests with all feature flags
test-all:
    cargo test --workspace --all-features

# Format code
fmt:
    cargo fmt -p zensally -p zensally-tract -p zensally-zentract -p zensally-depth -p zensally-micro

# Lint
clippy:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Check all feature permutations
feature-check:
    cargo check --workspace
    cargo check --workspace --features blazeface320
    cargo check --workspace --features mediapipe
    cargo check --workspace --features yunet
    cargo check --workspace --features analyzer
    cargo check --workspace --features "ultraface,microsalnet"
    cargo check --workspace --all-features

# Run benchmarks
bench *ARGS:
    cargo bench --package zensally-tract --features mediapipe --bench face_bench {{ARGS}}

# Full CI check (local)
ci: fmt
    just clippy
    just feature-check
    just test-all
    just bench -- --quick

# Download WIDER FACE validation dataset
download-wider-face:
    bash scripts/download_wider_face.sh

# Run WIDER FACE validation
validate:
    cargo run --package zensally-tract --example wider_validate --release

# Build the zentract ONNX inference plugin (.so/.dylib/.dll).
# The plugin is placed in zentract/target/release/ which zensally-zentract
# discovers automatically during development.
build-plugin:
    cargo build --release --manifest-path ../zentract/Cargo.toml -p zentract-abi
    @echo ""
    @echo "Plugin built:"
    @ls -lh ../zentract/target/release/libzentract_abi.so 2>/dev/null || \
     ls -lh ../zentract/target/release/libzentract_abi.dylib 2>/dev/null || \
     ls -lh ../zentract/target/release/zentract_abi.dll 2>/dev/null
    @echo ""
    @echo "zensally-zentract will find it automatically in the workspace."
    @echo "For deployment, copy it next to your binary or set ZENTRACT_PLUGIN_PATH."

# Build plugin and symlink into target/release/ for local dev
build-plugin-dev: build-plugin
    mkdir -p target/release
    ln -sf ../../../zentract/target/release/libzentract_abi.so target/release/ 2>/dev/null || true
    ln -sf ../../../zentract/target/release/libzentract_abi.dylib target/release/ 2>/dev/null || true

# Measure binary size of compressed models
model-sizes:
    @echo "ONNX model sizes:"
    @ls -la crates/zensally-tract/models/

# Regenerate the public-API surface snapshots (docs/public-api/)
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Development-only completed preprocessing profiler. No model or pixel changes.
coz-build:
    cargo build --release --manifest-path tools/coz-profile/Cargo.toml --locked

coz-check:
    cargo fmt --manifest-path tools/coz-profile/Cargo.toml -- --check
    cargo clippy --manifest-path tools/coz-profile/Cargo.toml --all-targets --locked -- -D warnings

coz-worker +args:
    ./tools/coz-profile/target/release/zensally-coz-profile {{args}}

coz-run profile +args:
    test ! -e "{{profile}}"
    {{env_var_or_default("COZ_BIN", "coz")}} run --source-scope '%/zensally/src/%' --output "{{profile}}" --- ./tools/coz-profile/target/release/zensally-coz-profile {{args}}

coz-core-test:
    cargo test --manifest-path tools/coz-profile/Cargo.toml --test core_unit --locked

# Dependency refresh checks. Keep debug artifacts small on shared build hosts.
verify-update:
    cargo fmt -p zensally -p zensally-tract -p zensally-zentract -p zensally-depth -p zensally-micro -- --check
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
    cargo test --workspace --all-features --locked
    cargo test --workspace --doc --all-features --locked
    cargo check -p zensally --no-default-features --locked
    cargo check -p zensally --no-default-features --features serde,zenlayout --locked

# Counts unique resolved normal/build dependencies, excluding dev tooling.
dependency-weight:
    python3 tools/dependency_weight.py

# 32-bit Linux validation (requires cross and Docker).
test-i686:
    CODEC_CORPUS_CACHE="{{env_var_or_default('CODEC_CORPUS_CACHE', '.external-cache')}}" cross test --workspace --all-features --target i686-unknown-linux-gnu

# Embedded-depth experiment, with current-main codec dependencies.
depth-check *CARGO_ARGS:
    cargo clippy {{CARGO_ARGS}} -p zensally-depth --all-targets --all-features -- -D warnings
    cargo test {{CARGO_ARGS}} -p zensally-depth --all-features
    cargo check {{CARGO_ARGS}} -p zensally-depth --no-default-features

# Refresh the dependency graph after a manifest/patch change.
deps-update *CARGO_ARGS:
    cargo {{CARGO_ARGS}} update

# Inspect the committed w24 graph without importing a training framework.
inspect-w24:
    python3 tools/inspect_microsalnet.py

# Regenerate the fixed plan, including validation of its supported ONNX subset.
export-w24:
    python3 tools/export_microsalnet.py

micro-check *CARGO_ARGS:
    python3 tools/export_microsalnet.py --check
    cargo clippy {{CARGO_ARGS}} -p zensally-micro --all-targets --all-features -- -D warnings
    cargo test {{CARGO_ARGS}} -p zensally-micro --all-features
    cargo test {{CARGO_ARGS}} -p zensally-micro --no-default-features
    cargo check {{CARGO_ARGS}} -p zensally-micro --no-default-features

micro-compare *CARGO_ARGS:
    cargo run {{CARGO_ARGS}} --release -p zensally-micro --features std --example compare_w24 -- compare

micro-bench *CARGO_ARGS:
    cargo bench {{CARGO_ARGS}} -p zensally-micro --features std --bench w24 -- --rounds=30 --cpu-set={{env_var("W24_CPU_SET")}}

micro-profile kind *CARGO_ARGS:
    cargo build {{CARGO_ARGS}} --release -p zensally-micro --features std --example compare_w24
    perf record -g -o "{{env_var('W24_PERF_OUTPUT')}}" -- "{{env_var('CARGO_TARGET_DIR')}}/release/examples/compare_w24" {{kind}} 200

depth-inspect input output *CARGO_ARGS:
    cargo run {{CARGO_ARGS}} -p zensally-depth --features heic --example inspect_heic_depth -- "{{input}}" "{{output}}"

depth-oracle input output:
    python3 tools/heic_depth_oracle.py "{{input}}" "{{output}}"

crop-rectangles corpus manifest report budget *CARGO_ARGS:
    cargo run {{CARGO_ARGS}} --release -p zensally-micro --features std --example eval_rectangles -- "{{corpus}}" "{{manifest}}" "{{report}}" "{{budget}}"

# Freeze validation-only corpus inputs with canonical source identities.
prepare-crop-corpus validation representatives output commit max_bytes:
    python3 tools/prepare_crop_corpus.py "{{validation}}" "{{representatives}}" "{{output}}" --source-commit "{{commit}}" --max-bytes {{max_bytes}}

# Frozen five-process confirmation, with strict zenbench admission.
micro-confirm *CARGO_ARGS:
    cargo bench {{CARGO_ARGS}} -p zensally-micro --features std --bench w24 -- --mode=confirm --processes=5 --rounds=30 --margin=1 --workload-id=w24-fixed256-v1 --cpu-set={{env_var("W24_CPU_SET")}}

# Resolve an independently maintained dependency workspace for review.
cohort-metadata manifest:
    cargo metadata --manifest-path "{{manifest}}" --format-version 1

# Reproduce the i686 embedded-depth regression with a writable container cache.
depth-i686:
    CODEC_CORPUS_CACHE=.external-cache cross test -p zensally-depth --all-features --target i686-unknown-linux-gnu --test heic_corpus

# Bare-metal checks prove default libraries do not need a standard library.
check-no-std-target target *CARGO_ARGS:
    cargo check {{CARGO_ARGS}} -p zensally-depth -p zensally-micro --no-default-features --target {{target}}

# Run an unfitted RGB/depth fusion experiment on a complete camera image.
depth-crops format input report:
    cargo run --release -p zensally-depth --features jpeg,heic --example depth_crops -- {{format}} "{{input}}" "{{report}}"

# Fail if a named hardware validation silently uses another execution tier.
micro-check-tier tier:
    ZENSALLY_REQUIRED_TIER={{tier}} cargo test -p zensally-micro --all-features -- --nocapture

# Separate deployment executables; RSS covers cold load plus first inference.
runtime-weight output commit:
    cargo build --release -p zensally-micro --features std --example weight_micro --example weight_tract
    python3 tools/measure_runtime_weight.py "{{env_var("CARGO_TARGET_DIR")}}/release/examples" "{{output}}" --source-commit "{{commit}}"

# Compare complete native depth-code captures, including orientation diagnostics.
depth-compare native oracle width height:
    python3 tools/compare_depth_planes.py "{{native}}" "{{oracle}}" {{width}} {{height}}

# Compare the frozen ordered runtime and candidate in one process.
reduction-check *ARGS:
    cargo test -p zensally-micro --features std,experimental-reassociate
    cargo test -p zensally-micro --features std,experimental-algebraic
    cargo test -p zensally-micro --features std,experimental-depthwise
    cargo clippy -p zensally-micro --all-targets --all-features -- -D warnings
    cargo clippy --manifest-path tools/reassociate-study/Cargo.toml --all-targets {{ARGS}} -- -D warnings

reduction-bench *ARGS:
    cargo bench --manifest-path tools/reassociate-study/Cargo.toml --bench reduction {{ARGS}} -- --rounds=30 --cpu-set={{env_var("W24_CPU_SET")}}

reduction-confirm *ARGS:
    cargo bench --manifest-path tools/reassociate-study/Cargo.toml --bench reduction {{ARGS}} -- --mode=confirm --processes=5 --rounds=30 --margin=1 --workload-id=w24-reduction256-v1 --cpu-set={{env_var("W24_CPU_SET")}}

# Independent fixed follow-up after broad intervals in the initial A/A control.
reduction-confirm-long *ARGS:
    cargo bench --manifest-path tools/reassociate-study/Cargo.toml --bench reduction {{ARGS}} -- --mode=confirm --processes=9 --rounds=100 --margin=1 --workload-id=w24-reduction256-v1 --cpu-set={{env_var("W24_CPU_SET")}}

# Invoke under run-heavy; sudo perf retains hardware-counter access.
reduction-perf executable arm output cpu repeats:
    test ! -e "{{output}}.stat"
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -o "{{output}}.stat" -- taskset -c {{cpu}} "{{executable}}" {{arm}} {{repeats}}

reduction-record executable arm output cpu repeats:
    test ! -e "{{output}}"
    sudo perf record -e cycles:u -F 499 -g --call-graph dwarf,8192 -o "{{output}}" -- taskset -c {{cpu}} "{{executable}}" {{arm}} {{repeats}}

reduction-memory executable arm output cpu repeats:
    test ! -e "{{output}}"
    sudo perf mem record -U -t load --ldlat=10 -o "{{output}}" -- taskset -c {{cpu}} "{{executable}}" {{arm}} {{repeats}}

reduction-summary artifacts output:
    python3 tools/reassociate-study/summarize.py "{{artifacts}}" "{{output}}"

# Ordered reuse/load-ahead kernels, exact operator oracles and persistent workers.
tiling-check:
    cargo test -p zensally-micro --features std,experimental-channel-tiles
    cargo test -p zensally-micro --features std,experimental-load-ahead
    cargo clippy -p zensally-micro --all-targets --all-features -- -D warnings
    cargo clippy --manifest-path tools/reassociate-study/Cargo.toml --all-targets --all-features -- -D warnings

# Each callback handles the same 32 images; workers are created before timing.
scaling-bench *ARGS:
    cargo bench --manifest-path tools/reassociate-study/Cargo.toml --bench scaling {{ARGS}} -- --mode=confirm --processes=5 --rounds=10 --margin=1 --workload-id=w24-batch32-v2 --cpu-set={{env_var("W24_CPU_SET")}}

# Linux study: enforce each requested process CPU set in a bounded root scope.
# Invoke through run-heavy; no resource-observation gates are changed.
scaling-grid *CPU_SETS:
    bash tools/reassociate-study/run_scaling.sh {{CPU_SETS}}

# Include the complete batch driver in counters; GNU time records its process RSS.
scaling-perf executable output cpus repeats arm="batch":
    test ! -e "{{output}}.stat"
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -o "{{output}}.stat" -- env W24_CPU_SET="{{cpus}}" /usr/bin/time -v -o "{{output}}.time" taskset -c "{{cpus}}" "{{executable}}" {{arm}} {{repeats}}

scaling-record executable output cpus repeats arm="batch":
    test ! -e "{{output}}"
    sudo perf record -e cycles:u -F 499 -g --call-graph dwarf,8192 -o "{{output}}" -- env W24_CPU_SET="{{cpus}}" taskset -c "{{cpus}}" "{{executable}}" {{arm}} {{repeats}}

tiling-summary artifacts data output:
    python3 tools/reassociate-study/tiling_summary.py "{{artifacts}}" "{{data}}" "{{output}}"

# Run under run-heavy; profile counts separately from uninstrumented timing.
allocation-record executable arm repeats output:
    test ! -e "{{output}}.zst"
    heaptrack --record-only -o "{{output}}" "{{executable}}" {{arm}} {{repeats}}

allocation-diff longer shorter output:
    test ! -e "{{output}}.txt"
    heaptrack_print -f "{{longer}}" -d "{{shorter}}" -n 20 -s 3 --flamegraph-cost-type allocations -F "{{output}}.stacks" > "{{output}}.txt"

allocation-summary stacks images output:
    python3 tools/reassociate-study/allocation_summary.py "{{stacks}}" {{images}} "{{output}}"

# Both sigmoid modes, independently linked old and current NN implementations.
memory-parity repeats:
    cargo run --release --manifest-path tools/reassociate-study/Cargo.toml -- parity {{repeats}}

memory-process executable repeats output cpu:
    test ! -e "{{output}}"
    /usr/bin/time -v -o "{{output}}" taskset -c "{{cpu}}" "{{executable}}" {{repeats}}

memory-perf executable repeats output cpu:
    test ! -e "{{output}}.stat"
    sudo perf stat --cputype=core -e cycles,instructions,task-clock,minor-faults,major-faults,context-switches,cpu-migrations -o "{{output}}.stat" -- /usr/bin/time -v -o "{{output}}.time" taskset -c "{{cpu}}" "{{executable}}" {{repeats}}

# Four standalone deployments, frozen parity and heaptrack count differentials.
# Invoke through run-heavy; output must be fresh and outside git.
memory-study output cpu="2":
    bash tools/reassociate-study/run_memory.sh "{{output}}" "{{cpu}}"

# i270 diagnostic counters; these do not establish timing confidence.
memory-counters executable output:
    bash tools/reassociate-study/run_profiles.sh "{{executable}}" "{{output}}"

memory-summary artifacts output:
    python3 tools/reassociate-study/memory_summary.py "{{artifacts}}" "{{output}}"

# Explicit source limits are for smoke checks; full evaluation runs every shard.
aspect-check:
    cargo test -p zensally-depth --features jpeg --example aspect_crops
    cargo clippy -p zensally-depth --features jpeg --example aspect_crops -- -D warnings
    cargo build --release -p zensally-depth --features jpeg --example aspect_crops

aspect-run executable dataset shards output:
    bash tools/aspect-study/run.sh "{{executable}}" "{{dataset}}" "{{shards}}" "{{output}}"

aspect-shards manifest output:
    python3 tools/aspect-study/shard.py "{{manifest}}" "{{output}}"

aspect-summary python run manifest output:
    "{{python}}" tools/aspect-study/summarize.py "{{run}}" "{{output}}" --manifest "{{manifest}}"

aspect-render-inputs manifest analysis output:
    python3 tools/aspect-study/render_inputs.py "{{manifest}}" "{{analysis}}" "{{output}}"

aspect-gallery python run analysis visuals:
    "{{python}}" tools/aspect-study/gallery.py "{{run}}" "{{analysis}}" "{{visuals}}"

aspect-oracle python dataset manifest gallery output:
    "{{python}}" tools/aspect-study/oracle.py "{{dataset}}" "{{manifest}}" "{{gallery}}" "{{output}}"

# Start the task-owned browser under run-heavy; CDP is internal automation only.
aspect-browser chromium profile:
    mkdir -p "{{profile}}/scratch"
    TMPDIR="{{profile}}/scratch" "{{chromium}}" --no-sandbox --disable-dev-shm-usage --user-data-dir="{{profile}}/state" --remote-debugging-address=127.0.0.1 --remote-debugging-port=3993 --window-size=1600,1300 about:blank

aspect-browser-check url output endpoint="http://127.0.0.1:3993":
    node tools/aspect-study/browser_check.mjs "{{url}}" "{{output}}" "{{endpoint}}"
