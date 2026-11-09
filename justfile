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
depth-corpus-prefetch:
    CODEC_CORPUS_CACHE=.external-cache cargo test -p zensally-depth --all-features --test heic_corpus

depth-i686: depth-corpus-prefetch
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

aspect-browser-check url output endpoint="http://127.0.0.1:3993" methods="8":
    node tools/aspect-study/browser_check.mjs "{{url}}" "{{output}}" "{{endpoint}}" "{{methods}}"

aspect-compare original corrected output:
    python3 tools/aspect-study/compare.py "{{original}}" "{{corrected}}" "{{output}}"

aspect-placement-oracle render output:
    python3 tools/aspect-study/placement_oracle.py "{{render}}" "{{output}}"

# Research-only geometry; no torch dependency for the policy guards.
crop-first-policy-check:
    PYTHONPATH=training python3 -m unittest discover -s training -p 'test_crop_policy.py' -v

# Explicitly requires the existing training environment's torch.
crop-first-model-check:
    PYTHONPATH=training python3 -m unittest discover -s training -p 'test_crop_ranker.py' -v

crop-first-smoke output:
    python3 tools/crop-first/smoke_model.py "{{output}}"

crop-first-inventory manifests output:
    python3 tools/crop-first/inventory.py "{{manifests}}" "{{output}}"

crop-first-teacher-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p 'test_teacher.py' -v

crop-first-teacher-prepare task output purpose:
    python3 tools/crop-first/teacher.py prepare "{{task}}" "{{output}}" --purpose "{{purpose}}"

crop-first-render-check:
    cargo clippy -p zensally-depth --features jpeg --example crop_label_pairs -- -D warnings
    cargo build --release -p zensally-depth --features jpeg --example crop_label_pairs

crop-first-render executable source output source_id proposals:
    "{{executable}}" "{{source}}" "{{output}}" "{{source_id}}" "{{proposals}}"

crop-first-pilot manifests executable inputs output quarantine:
    python3 tools/crop-first/pilot.py "{{manifests}}" "{{executable}}" "{{inputs}}" "{{output}}" "{{quarantine}}"

crop-first-cli-capture rollout tasks output:
    python3 tools/crop-first/capture_cli.py "{{rollout}}" "{{tasks}}" "{{output}}"

crop-first-gallery pilot:
    python3 tools/crop-first/gallery.py "{{pilot}}"

crop-first-artifacts action directory:
    python3 tools/crop-first/artifacts.py "{{action}}" "{{directory}}"


crop-first-proposal-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p 'test_proposals.py' -v

crop-first-pilot-reuse manifests executable inputs output quarantine proposals:
    python3 tools/crop-first/pilot.py "{{manifests}}" "{{executable}}" "{{inputs}}" "{{output}}" "{{quarantine}}" --reuse-inputs --vision-proposals "{{proposals}}"

crop-first-usage rollout output:
    python3 tools/crop-first/usage.py "{{rollout}}" "{{output}}"


crop-preview-build:
    cargo update --manifest-path tools/crop-preview/Cargo.toml
    cargo clippy --manifest-path tools/crop-preview/Cargo.toml -- -D warnings
    cargo build --release --manifest-path tools/crop-preview/Cargo.toml

crop-preview-check executable:
    python3 tools/crop-first/check_preview.py "{{executable}}"

crop-expanded-freeze manifests quarantine inputs output:
    python3 tools/crop-first/expanded.py freeze "{{inputs}}" "{{output}}" --manifests "{{manifests}}" --quarantine "{{quarantine}}"

crop-expanded-resolve inputs output variant_root:
    python3 tools/crop-first/expanded.py resolve "{{inputs}}" "{{output}}" --variant-root "{{variant_root}}"

crop-expanded-acquire python inputs output executable:
    "{{python}}" tools/crop-first/expanded.py acquire "{{inputs}}" "{{output}}" --executable "{{executable}}"

crop-expanded-render python inputs output executable batch proposals:
    "{{python}}" tools/crop-first/expanded.py render "{{inputs}}" "{{output}}" --executable "{{executable}}" --batch "{{batch}}" --proposals "{{proposals}}"

crop-expanded-capture rollout batch output:
    python3 tools/crop-first/capture_proposals.py "{{rollout}}" "{{batch}}" "{{output}}"

crop-token-compare cases output:
    python3 tools/crop-first/token_compare.py "{{cases}}" "{{output}}"

crop-expanded-send kind batch:
    python3 tools/crop-first/batch.py "{{kind}}" "{{batch}}"

crop-expanded-localize batch root:
    python3 tools/crop-first/batch.py localize "{{batch}}" --root "{{root}}"

crop-expanded-source-oracle python inputs executable output:
    "{{python}}" tools/crop-first/source_oracle.py "{{inputs}}" "{{executable}}" "{{output}}"

crop-expanded-status inputs output:
    python3 tools/crop-first/expanded.py status "{{inputs}}" "{{output}}"

crop-expanded-render-many python inputs output executable batches:
    "{{python}}" tools/crop-first/expanded.py render_many "{{inputs}}" "{{output}}" --executable "{{executable}}" --batch-indices "{{batches}}"

crop-expanded-fetch batch root remote remote_root:
    python3 tools/crop-first/batch.py fetch "{{batch}}" --root "{{root}}" --remote "{{remote}}" --remote-root "{{remote_root}}"

crop-expanded-run root rollout batches:
    python3 tools/crop-first/run_expanded.py "{{root}}" "{{rollout}}" "{{batches}}" --execute

crop-expanded-plan root rollout batches:
    python3 tools/crop-first/run_expanded.py "{{root}}" "{{rollout}}" "{{batches}}"

crop-expanded-report inputs output:
    python3 tools/crop-first/aggregate.py "{{inputs}}" "{{output}}"

crop-expanded-report-partial inputs output:
    python3 tools/crop-first/aggregate.py "{{inputs}}" "{{output}}" --partial

crop-expanded-gallery output:
    python3 tools/crop-first/gallery.py "{{output}}" --expanded

crop-cli-capture-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p 'test_capture_cli.py' -v

crop-source-lfs-check variant_root resolution sources output:
    python3 tools/crop-first/check_lfs.py "{{variant_root}}" "{{resolution}}" "{{sources}}" "{{output}}"

crop-expanded-gallery-partial output count:
    python3 tools/crop-first/gallery.py "{{output}}" --expanded --labels "cli-labels-partial-{{count}}.json" --output "review-partial-{{count}}.html"

crop-coverage-audit python inputs output executable:
    "{{python}}" tools/crop-first/coverage_audit.py "{{inputs}}" "{{output}}" "{{executable}}"

crop-coverage-report inputs output:
    python3 tools/crop-first/aggregate.py "{{inputs}}" "{{output}}" --coverage-audit

crop-gallery-browser output chromium scratch:
    python3 tools/crop-first/browser_check.py "{{output}}" "{{chromium}}" "{{scratch}}"

crop-cost-summary output rollout:
    python3 tools/crop-first/cost_summary.py "{{output}}" "{{rollout}}"

crop-train-probe python output *args:
    "{{python}}" tools/crop-first/train_probe.py "{{output}}" {{args}}

crop-regions-check python:
    PYTHONPATH=training "{{python}}" -m unittest discover -s training -p test_crop_regions.py -v

crop-train-matrix python plan output:
    python3 tools/crop-first/train_matrix.py "{{python}}" "{{plan}}" "{{output}}"

crop-direct-zones task cache output:
    python3 tools/crop-first/direct_zones.py "{{task}}" "{{cache}}" "{{output}}"

crop-direct-zones-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p test_direct_zones.py -v

crop-direct-render python packets inputs output renderer:
    "{{python}}" tools/crop-first/render_direct.py "{{packets}}" "{{inputs}}" "{{output}}" "{{renderer}}"

crop-preview-oracle python output renderer report:
    "{{python}}" tools/crop-first/preview_oracle.py "{{output}}" "{{renderer}}" "{{report}}"

crop-direct-browser output chromium scratch:
    python3 tools/crop-first/browser_check.py "{{output}}" "{{chromium}}" "{{scratch}}" --suite direct

crop-pool-numeric python output:
    "{{python}}" tools/crop-first/pool_numeric.py "{{output}}"

# Current family-map reconciliation, with historical evaluation exclusions.
crop-corpus-freeze output:
    python3 tools/crop-first/corpus.py "{{output}}"

crop-corpus-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p test_corpus.py -v

crop-photo-audit corpus output:
    python3 tools/crop-first/photo_audit.py "{{corpus}}" "{{output}}"

crop-photo-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p test_photo_focus.py -v

crop-sunburst-costs output:
    python3 tools/crop-first/sunburst_costs.py "{{output}}"

crop-sunburst-generate plan cache output *args:
    python3 tools/crop-first/sunburst_generate.py "{{plan}}" "{{cache}}" "{{output}}" {{args}}

crop-sunburst-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p test_sunburst.py -v

crop-sunburst-calculator-oracle captured_js output:
    python3 tools/crop-first/sunburst_calculator_check.py "{{captured_js}}" "{{output}}"

crop-sunburst-gallery output:
    python3 tools/crop-first/sunburst_gallery.py "{{output}}"

crop-sunburst-browser output chromium scratch:
    python3 tools/crop-first/browser_check.py "{{output}}" "{{chromium}}" "{{scratch}}" --suite sunburst

crop-decode-build:
    cargo fmt --manifest-path tools/crop-decode/Cargo.toml --check
    cargo clippy --manifest-path tools/crop-decode/Cargo.toml -- -D warnings
    cargo build --release --manifest-path tools/crop-decode/Cargo.toml

crop-current-acquire corpus output decoder renderer resizer:
    python3 tools/crop-first/acquire_current.py "{{corpus}}" "{{output}}" "{{decoder}}" "{{renderer}}" "{{resizer}}"

crop-fit-plan inputs output *args:
    python3 tools/crop-first/fit_study.py zones "{{inputs}}" "{{output}}" {{args}}

crop-fit-render inputs output results cache renderer:
    python3 tools/crop-first/fit_study.py render "{{inputs}}" "{{output}}" --results "{{results}}" --cache "{{cache}}" --renderer "{{renderer}}"

crop-fit-api plan cache output workers="4":
    python3 tools/crop-first/api_batch.py "{{plan}}" "{{cache}}" "{{output}}" --workers {{workers}}

crop-review-check:
    PYTHONPATH=tools/crop-first python3 -m unittest discover -s tools/crop-first -p test_review.py -v

crop-fit-check python:
    PYTHONPATH=training "{{python}}" -m unittest discover -s training -p test_crop_fit.py -v

crop-fit-admit rendered results cache output:
    python3 tools/crop-first/admit.py "{{rendered}}" "{{results}}" "{{cache}}" "{{output}}"

crop-fit python manifest output *args:
    "{{python}}" tools/crop-first/fit.py "{{manifest}}" "{{output}}" {{args}}

crop-fit-resume-check python output *args:
    "{{python}}" tools/crop-first/check_fit_resume.py "{{output}}" {{args}}

crop-fit-gallery rendered output *args:
    python3 tools/crop-first/fit_gallery.py "{{rendered}}" "{{output}}" {{args}}

crop-fit-browser output chromium scratch:
    python3 tools/crop-first/browser_check.py "{{output}}" "{{chromium}}" "{{scratch}}" --suite fit

crop-fit-profile python manifest output *args:
    "{{python}}" tools/crop-first/fit_profile.py "{{manifest}}" "{{output}}" {{args}}

crop-api-ledger cache output:
    python3 tools/crop-first/api_ledger.py "{{cache}}" "{{output}}"

crop-fit-policy-report run output:
    python3 tools/crop-first/fit_policy_report.py "{{run}}" "{{output}}"

crop-acquisition-oracle inputs output oracle:
    python3 tools/crop-first/acquisition_oracle.py "{{inputs}}" "{{output}}" --jxl-oracle "{{oracle}}"

crop-train-pair zenbench plan output:
    "{{zenbench}}" pair "{{plan}}" --output "{{output}}"

crop-train-nsys python output:
    mkdir -p "{{output}}"
    NSYS_TMPDIR="${TMPDIR:?set disk-backed TMPDIR}" sudo -n --preserve-env=NSYS_TMPDIR,TMPDIR,TORCHINDUCTOR_COMPILE_THREADS,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR,OMP_NUM_THREADS,MKL_NUM_THREADS /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,cudnn --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-devices=0 --output="{{output}}/nsight" "{{python}}" tools/crop-first/train_probe.py "{{output}}/probe" --batch 64 --precision bf16 --channels-last --optimizer fused --compile default --steps 1000 --nsys

# Offline storage codecs use separate dependency cohorts.
crop-storage-build:
    cargo build --release --manifest-path tools/crop-compress/Cargo.toml
    cargo build --release --manifest-path tools/crop-png/Cargo.toml

crop-storage-clippy:
    cargo clippy --manifest-path tools/crop-compress/Cargo.toml --all-targets -- -D warnings
    cargo clippy --manifest-path tools/crop-png/Cargo.toml --all-targets -- -D warnings

crop-sunburst-commerce-plan output:
    python3 tools/crop-first/sunburst_commerce_v2.py {{output}}

crop-sunburst-recompress source output png_reader encoder decoder:
    python3 tools/crop-first/sunburst_recompress.py {{source}} {{output}} --png-reader {{png_reader}} --encoder {{encoder}} --decoder {{decoder}}

crop-sunburst-dataset source comparison dataset collection native_archive:
    python3 tools/crop-first/sunburst_dataset.py {{source}} {{comparison}} {{dataset}} {{collection}} --native-archive {{native_archive}}

crop-sunburst-status root worker commit:
    python3 tools/crop-first/sunburst_status.py {{root}} --worker {{worker}} --build-commit {{commit}} --write

# Read only completed native files from the API host; preserve local inspection notes.
crop-sunburst-collect remote local:
    rsync -rt --include='*.png' --include='*.json' --exclude='*' {{remote}}/ {{local}}/

crop-sunburst-verify root cache output carry:
    python3 tools/crop-first/sunburst_verify.py {{root}} {{cache}} {{output}} --carry-journal {{carry}}

crop-sunburst-products-selfies-plan output:
    python3 tools/crop-first/sunburst_products_selfies_v3.py {{output}}

# Pure local evaluation; neither path imports model APIs or accesses credentials.
crop-cached-eval manifest selections output use *args:
    python3 tools/crop-first/cached_crop_eval.py {{manifest}} {{selections}} {{output}} --use {{use}} {{args}}

crop-cached-eval-check:
    python3 -m unittest discover -s tools/crop-first -p test_cached_crop_eval.py

# Metadata only; explicit Python supplies flatbuffers 25.12.19 / tflite 2.18.0.
crop-vision-inventory python root *args:
    {{python}} tools/crop-first/vision_inventory.py {{root}} {{args}}

# Existing-model geometry evaluation; raw RGB inputs are hashed by the manifest tool.
runtime-reference python mode manifest model output *args:
    {{python}} tools/runtime-eval/mediapipe_probe.py {{mode}} {{manifest}} {{model}} {{output}} {{args}}

runtime-rust-build:
    cargo build --release -p zensally-tract --all-features --example runtime_probe

runtime-rust mode model manifest output repeats:
    target/release/examples/runtime_probe {{mode}} {{model}} {{manifest}} {{output}} {{repeats}}

runtime-prepare python output:
    {{python}} tools/runtime-eval/prepare.py {{output}}

runtime-evaluate python manifest output *args:
    {{python}} tools/runtime-eval/run_models.py {{manifest}} {{output}} {{args}}

runtime-plan python manifest output *args:
    {{python}} tools/runtime-eval/plan.py {{manifest}} {{output}} --python {{python}} {{args}}

runtime-gallery python root:
    {{python}} tools/runtime-eval/gallery.py {{root}}

runtime-pair zenbench plan output:
    sudo systemd-run --scope --quiet -p AllowedCPUs=2 -p MemoryMax=4G -p MemorySwapMax=0 sudo -u lilith env TMPDIR=/home/lilith/tmp {{zenbench}} pair {{plan}} --output {{output}}

runtime-face-audit root output:
    python3 tools/runtime-eval/face_audit.py {{root}} {{output}}

runtime-tile-reference python mode manifest model output *args:
    {{python}} tools/runtime-eval/tile_probe.py {{mode}} {{manifest}} {{model}} {{output}} {{args}}

runtime-profile root python binary:
    bash tools/runtime-eval/profile.sh {{root}} {{python}} {{binary}}

runtime-followup-plan root binary:
    python3 tools/runtime-eval/followup_plan.py {{root}} {{binary}}

runtime-tflite-smoke model:
    cargo run --release --manifest-path tools/runtime-eval/native/Cargo.toml -- {{model}}

composition-acquire output:
    python3 tools/composition/acquire_teachers.py {{output}}

composition-prepare python root preview:
    {{python}} tools/composition/prepare.py {{root}} --preview {{preview}}

composition-geometry python root:
    {{python}} tools/composition/geometry.py {{root}}

composition-attention python kind root output *args:
    {{python}} tools/composition/attention.py {{kind}} {{root}} {{output}} {{args}}

composition-cache python root checkpoint:
    {{python}} tools/composition/cache_features.py {{root}} --checkpoint {{checkpoint}}

composition-feature-check python:
    {{python}} -m unittest discover -s training -p 'test_composition_*.py' -v

composition-fit-heads python root:
    {{python}} tools/composition/fit_heads.py {{root}}

composition-analyze python root:
    {{python}} tools/composition/analyze_heads.py {{root}}

composition-recover-predictions python root:
    {{python}} tools/composition/recover_predictions.py {{root}}

composition-witness root output:
    python3 tools/composition/witness.py {{root}} {{output}}

composition-witness-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite composition

composition-acquire-critic python root:
    {{python}} tools/composition/acquire_critic.py {{root}}

composition-prepare-critic root preview:
    python3 tools/composition/prepare_critic.py {{root}} {{preview}}

composition-critic python model plan output *args:
    {{python}} tools/composition/critic_probe.py {{model}} {{plan}} {{output}} {{args}}

composition-prepare-auxiliary python root output preview:
    {{python}} tools/composition/prepare_auxiliary.py {{root}} {{output}} {{preview}}

composition-fit-joint python root first_gate checkpoint:
    {{python}} tools/composition/fit_joint.py {{root}} {{first_gate}} {{checkpoint}}

composition-acquire-croppers python output:
    {{python}} tools/composition/acquire_croppers.py {{output}}

composition-focalnet python models first_gate auxiliary output:
    {{python}} tools/composition/focalnet_probe.py {{models}} {{first_gate}} {{auxiliary}} {{output}}

composition-focalnet-bank python models first_gate auxiliary output:
    {{python}} tools/composition/focalnet_bank.py {{models}} {{first_gate}} {{auxiliary}} {{output}}

composition-framelite python models first_gate preview output:
    {{python}} tools/composition/framelite_probe.py {{models}} {{first_gate}} {{preview}} {{output}}

composition-framelite-ratios python models first_gate output:
    {{python}} tools/composition/framelite_ratios.py {{models}} {{first_gate}} {{output}}

composition-focalnet-bench python model cache *args:
    {{python}} tools/composition/focalnet_bench.py {{model}} {{cache}} {{args}}

composition-pack python first_gate auxiliary output:
    {{python}} tools/composition/pack_evidence.py {{first_gate}} {{auxiliary}} {{output}}

composition-crop-gallery models output *args:
    python3 tools/composition/crop_gallery.py {{models}} {{output}} {{args}}

composition-crop-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite local-crops

composition-semantic-witness first_gate output:
    python3 tools/composition/semantic_witness.py {{first_gate}} {{output}}

composition-semantic-bank python models auxiliary annotations output:
    {{python}} tools/composition/semantic_bank.py {{models}} {{auxiliary}} {{annotations}} {{output}}

composition-pool-numerics python first_gate checkpoint output:
    {{python}} tools/composition/check_pool_numerics.py {{first_gate}} {{checkpoint}} {{output}}

composition-localizer-plan first_gate output:
    python3 tools/composition/localizer_plan.py {{first_gate}} {{output}}

composition-localizer python root first_gate preview output *args:
    {{python}} tools/composition/localizer_probe.py {{root}} {{first_gate}} {{preview}} {{output}} {{args}}

composition-cache-extents python root auxiliary checkpoint:
    {{python}} tools/composition/cache_extents.py {{root}} {{auxiliary}} {{checkpoint}}

composition-fit-extents python root *args:
    {{python}} tools/composition/fit_extents.py {{root}} {{args}}

composition-fit-image-extents python root auxiliary checkpoint:
    {{python}} tools/composition/fit_extent_images.py {{root}} {{auxiliary}} {{checkpoint}}

composition-cache-focal-extents python models root auxiliary output:
    {{python}} tools/composition/cache_focal_extents.py {{models}} {{root}} {{auxiliary}} {{output}}

composition-analyze-extents root output *args:
    python3 tools/composition/analyze_extents.py {{root}} {{output}} {{args}}

composition-fit-extent-hints python root auxiliary checkpoint:
    {{python}} tools/composition/fit_extent_hints.py {{root}} {{auxiliary}} {{checkpoint}}

composition-audit-feasibility bank output:
    python3 tools/composition/audit_crop_feasibility.py {{bank}} {{output}}

composition-constrained-bank python models auxiliary output:
    {{python}} tools/composition/constrained_bank.py {{models}} {{auxiliary}} {{output}}

composition-pack-successor python models localizer auxiliary output:
    {{python}} tools/composition/pack_successor.py {{models}} {{localizer}} {{auxiliary}} {{output}}

crop-artifacts action directory:
    python3 tools/crop-first/artifacts.py {{action}} {{directory}}

composition-region-retention annotations bank output admission:
    python3 tools/composition/region_retention.py {{annotations}} {{bank}} {{output}} --admission {{admission}}

composition-split-focalnet python model output:
    {{python}} tools/composition/split_focalnet.py {{model}} {{output}}

composition-verify-cached python models split auxiliary bank output *args:
    {{python}} tools/composition/verify_cached_focalnet.py {{models}} {{split}} {{auxiliary}} {{bank}} {{output}} {{args}}

composition-cached-bench python models split cache bank *args:
    {{python}} tools/composition/cached_focalnet_bench.py {{models}} {{split}} {{cache}} {{bank}} {{args}}

composition-cached-plan models split cache bank output python *args:
    python3 tools/composition/cached_timing_plan.py {{models}} {{split}} {{cache}} {{bank}} {{output}} --python {{python}} {{args}}

# Arrow Lake performance-core event definitions; this study uses CPUs 2–5.
composition-cached-profile python models split cache bank output threads query_threads *args:
    sudo perf stat -o {{output}}.stat -e 'cpu_core/event=0xa4,umask=0x1/,cpu_core/event=0xa4,umask=0x2/,cpu_core/event=0xa4,umask=0x10/,cpu_core/cycles/,cpu_core/instructions/,cpu_core/event=0xd1,umask=0x20/' -- perf record -q -g -F 499 -o {{output}}.data -- sudo -u "$USER" env TMPDIR="$HOME/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v -o {{output}}.time taskset -c 2,3,4,5 {{python}} tools/composition/cached_focalnet_bench.py {{models}} {{split}} {{cache}} {{bank}} --backend cached --threads {{threads}} --query-threads {{query_threads}} --repeats 60 {{args}}

composition-cached-summary root zenbench output *studies:
    python3 tools/composition/summarize_cached_runtime.py {{root}} {{zenbench}} --output {{output}} --studies {{studies}}

composition-query-head python query output:
    {{python}} tools/composition/extract_query_head.py {{query}} {{output}}

composition-snapshot-baker source output:
    python3 tools/composition/snapshot_owned_baker.py {{source}} {{output}}

composition-query-fixture python split auxiliary bank output *args:
    {{python}} tools/composition/prepare_query_fixture.py {{split}} {{auxiliary}} {{bank}} {{output}} {{args}}

composition-native-query-gate python binary fixture output:
    {{python}} tools/composition/verify_native_query.py {{binary}} {{fixture}} {{output}}

composition-native-query-build head baker target:
    ZENSALLY_QUERY_HEAD={{head}} CARGO_TARGET_DIR={{target}} cargo build --config 'patch.crates-io.zenpredict-bake.path="{{baker}}"' --release --manifest-path tools/query-native/Cargo.toml

composition-native-query-check head baker target:
    ZENSALLY_QUERY_HEAD={{head}} CARGO_TARGET_DIR={{target}} cargo clippy --config 'patch.crates-io.zenpredict-bake.path="{{baker}}"' --release --manifest-path tools/query-native/Cargo.toml -- -D warnings

composition-query-phase python split fixture *args:
    {{python}} tools/composition/query_phase_bench.py {{split}} {{fixture}} {{args}}

composition-query-timing-plan split fixture dispatched scalar output python *args:
    python3 tools/composition/query_timing_plan.py {{split}} {{fixture}} {{dispatched}} {{scalar}} {{output}} --python {{python}} {{args}}

composition-native-query-heap binary fixture output repeats math:
    heaptrack --record-only --output {{output}} {{binary}} bench {{fixture}} {{repeats}} {{math}}

# Arrow Lake performance core; pass another CPU when profiling another host.
composition-native-query-profile binary fixture output repeats math cpu:
    sudo perf stat -o {{output}}.stat -e 'cpu_core/event=0xa4,umask=0x1/,cpu_core/event=0xa4,umask=0x2/,cpu_core/event=0xa4,umask=0x10/,cpu_core/cycles/,cpu_core/instructions/,cpu_core/event=0xd1,umask=0x20/' -- perf record -q -B -N --no-bpf-event --synth=no -g -F 499 -e 'cpu_core/cycles/' -o {{output}}.data -- sudo -u "$USER" env TMPDIR="$HOME/tmp" /usr/bin/time -v -o {{output}}.time taskset -c {{cpu}} {{binary}} bench {{fixture}} {{repeats}} {{math}}

composition-mirror source destination:
    rsync -rt --mkpath --info=stats2 {{source}} {{destination}}

composition-pack-cached python runtime zenbench output:
    {{python}} tools/composition/pack_cached_runtime.py {{runtime}} {{zenbench}} {{output}}

composition-focal-preferences-cache python models first_gate output *args:
    {{python}} tools/composition/cache_focal_preferences.py {{models}} {{first_gate}} {{output}} {{args}}

composition-focal-preferences-fit python cache output *args:
    {{python}} tools/composition/fit_focal_preferences.py {{cache}} {{output}} {{args}}

composition-focal-preferences-analysis python cache fits output *args:
    {{python}} tools/composition/analyze_focal_preferences.py {{cache}} {{fits}} {{output}} {{args}}

composition-photo-preferences-cohort python cache rendered output:
    {{python}} tools/composition/photo_preference_cohort.py {{cache}} {{rendered}} {{output}}

composition-preference-gallery rendered cohort photo_fits mixed_fits output asset_base:
    python3 tools/composition/preference_gallery.py {{rendered}} {{cohort}} {{photo_fits}} {{mixed_fits}} {{output}} --asset-base {{asset_base}}

composition-pack-preferences python mixed photo rendered output:
    {{python}} tools/composition/pack_preference_evidence.py {{mixed}} {{photo}} {{rendered}} {{output}}

composition-owned-photo-plan audit corpus output:
    python3 tools/composition/owned_photo_plan.py {{audit}} {{corpus}} {{output}}

composition-prepare-owned-screen python acquired plan oracle output:
    {{python}} tools/composition/prepare_owned_screen.py {{acquired}} {{plan}} {{oracle}} {{output}}

composition-verify-owned-pixels python acquired output:
    {{python}} tools/composition/verify_owned_pixels.py {{acquired}} {{output}}

composition-owned-photo-proposals python root models split output *args:
    {{python}} tools/composition/owned_photo_proposals.py {{root}} {{models}} {{split}} {{output}} {{args}}

composition-prepare-pose-heads frozen_screen output:
    python3 tools/composition/prepare_pose_heads.py {{frozen_screen}} {{output}}

composition-pose-head-unit:
    python3 -m unittest discover -s tools/composition -p test_pose_head_bounds.py

composition-pose-head-review screen acquired experiment resizer encoder output:
    python3 tools/composition/pose_head_review.py {{screen}} {{acquired}} {{experiment}} {{resizer}} {{encoder}} {{output}}

composition-pose-head-readback python review:
    {{python}} tools/composition/verify_pose_head_review.py {{review}}

composition-pose-head-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite pose-head-review

composition-freeze-pose-heads root:
    python3 tools/composition/freeze_pose_head_screen.py {{root}}

composition-person-region-audit root output:
    python3 tools/composition/person_region_audit.py {{root}} {{output}}

composition-owned-screen-gallery root output asset_base:
    python3 tools/composition/owned_screen_gallery.py {{root}} {{output}} --asset-base {{asset_base}}

composition-svg-gutter-check python report output *args:
    {{python}} tools/composition/verify_svg_gutters.py {{report}} {{output}} {{args}}

composition-person-rois screen acquired runtime_manifest resizer encoder output:
    python3 tools/composition/prepare_person_rois.py {{screen}} {{acquired}} {{runtime_manifest}} {{resizer}} {{encoder}} {{output}}

composition-person-roi-probe python manifest output *args:
    {{python}} tools/composition/probe_person_rois.py {{manifest}} {{output}} {{args}}

composition-roi-unit:
    python3 -m unittest discover -s tools/composition -p test_roi_geometry.py

composition-person-roi-oracle python manifest output *args:
    {{python}} tools/composition/verify_person_rois.py {{manifest}} {{output}} {{args}}

composition-synthetic-person-rois runtime roles resizer encoder output:
    python3 tools/composition/prepare_synthetic_person_rois.py {{runtime}} {{roles}} {{resizer}} {{encoder}} {{output}}

composition-synthetic-roi-suitability root:
    python3 tools/composition/analyze_synthetic_rois.py {{root}}

composition-cache-roi-heads python root runtime models resizer output:
    {{python}} tools/composition/cache_roi_heads.py {{root}} {{runtime}} {{models}} {{resizer}} {{output}}

composition-cache-small-roi-heads python cache checkpoint output:
    {{python}} tools/composition/cache_small_roi_heads.py {{cache}} {{checkpoint}} {{output}}

composition-roi-head-unit python:
    {{python}} -m unittest discover -s training -p test_roi_head_fit.py

composition-fit-roi-heads python cache output:
    {{python}} tools/composition/fit_roi_heads.py {{cache}} {{output}}

composition-analyze-roi-heads fits output:
    python3 tools/composition/analyze_roi_heads.py {{fits}} {{output}}

composition-compare-roi-encoders focal small:
    python3 tools/composition/compare_roi_encoders.py {{focal}} {{small}}

composition-pack-roi-heads synthetic small output:
    python3 tools/composition/pack_roi_head_evidence.py {{synthetic}} {{small}} {{output}}

composition-unsplash-inventory checkout remote corpus output *args:
    python3 tools/composition/inventory_unsplash.py {{checkout}} {{remote}} {{corpus}} {{output}} {{args}}

composition-corpus-preview-build target:
    CARGO_TARGET_DIR={{target}} cargo build --locked --release --manifest-path tools/corpus-preview/Cargo.toml

composition-corpus-preview-clippy target:
    CARGO_TARGET_DIR={{target}} cargo clippy --locked --manifest-path tools/corpus-preview/Cargo.toml -- -D warnings

composition-fingerprint-check target:
    CARGO_TARGET_DIR={{target}} cargo test --locked --manifest-path tools/corpus-preview/Cargo.toml --bin corpus-fingerprint
    CARGO_TARGET_DIR={{target}} cargo clippy --locked --manifest-path tools/corpus-preview/Cargo.toml --bin corpus-fingerprint -- -D warnings
    CARGO_TARGET_DIR={{target}} cargo build --locked --release --manifest-path tools/corpus-preview/Cargo.toml --bin corpus-fingerprint

composition-unsplash-relationships previews binary output:
    python3 tools/composition/audit_unsplash_relationships.py {{previews}} {{binary}} {{output}}

composition-unsplash-freeze-roles previews relationships detectors output:
    python3 tools/composition/freeze_unsplash_roles.py {{previews}} {{relationships}} {{detectors}} {{output}}

composition-cache-unsplash-ranks python registry models split resizer output *args:
    {{python}} tools/composition/cache_unsplash_ranks.py {{registry}} {{models}} {{split}} {{resizer}} {{output}} {{args}}

composition-rank-target-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_rank_targets.py

composition-local-rank-unit python:
    {{python}} -m unittest discover -s training -p test_local_rank_distillation.py

composition-fit-unsplash-ranks python cache checkpoint output *args:
    {{python}} tools/composition/fit_unsplash_ranks.py {{cache}} {{checkpoint}} {{output}} {{args}}

composition-analyze-unsplash-ranks fits registry output:
    python3 tools/composition/analyze_unsplash_ranks.py {{fits}} {{registry}} {{output}}

composition-evaluate-unsplash-rank-test python cache plan output:
    {{python}} tools/composition/evaluate_unsplash_rank_test.py {{cache}} {{plan}} {{output}}

composition-profile-unsplash-training python cache checkpoint output *args:
    bash tools/composition/profile_unsplash_training.sh {{python}} {{cache}} {{checkpoint}} {{output}} {{args}}

composition-export-unsplash-ranker python fit_cache test_cache plan output *args:
    {{python}} tools/composition/export_unsplash_ranker.py {{fit_cache}} {{test_cache}} {{plan}} {{output}} {{args}}

composition-freeze-unsplash-ranks output_base:
    python3 tools/composition/freeze_unsplash_rank_evidence.py {{output_base}}

composition-qualify-graphed-rank python cache plan output:
    {{python}} tools/composition/qualify_graphed_rank_query.py {{cache}} {{plan}} {{output}}

composition-graphed-rank-plan python cache weights output:
    python3 tools/composition/graphed_rank_timing_plan.py {{python}} {{cache}} {{weights}} {{output}}

composition-verify-packet root:
    python3 tools/crop-first/artifacts.py verify {{root}}

composition-verify-remote-packet host root:
    ssh {{host}} nice -n19 ionice -c3 python3 - verify {{root}} < tools/crop-first/artifacts.py

composition-student-runtime-plan python cache weights export output *args:
    python3 tools/composition/student_runtime_timing_plan.py {{python}} {{cache}} {{weights}} {{export}} {{output}} {{args}}

composition-unsplash-rank-gallery python cache registry fits output:
    {{python}} tools/composition/unsplash_rank_gallery.py {{cache}} {{registry}} {{fits}} {{output}}

composition-profile-student-runtime python cache weights export output:
    bash tools/composition/profile_student_runtime.sh {{python}} {{cache}} {{weights}} {{export}} {{output}}

composition-unsplash-rank-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite unsplash-rank

composition-profile-student-onnx python cache export review output:
    {{python}} tools/composition/profile_student_onnx.py {{cache}} {{export}} {{review}} {{output}}

composition-pack-student-runtime base output:
    python3 tools/composition/pack_student_runtime_evidence.py {{base}} {{output}}

composition-export-student-channel-gather python fit_cache test_cache plan output:
    {{python}} tools/composition/export_student_channel_gather.py {{fit_cache}} {{test_cache}} {{plan}} {{output}} --gather-pool

composition-qualify-student-batching python study export output:
    {{python}} tools/composition/qualify_student_batching.py {{study}} {{export}} {{output}}

composition-summarize-student-runtime base output:
    python3 tools/composition/summarize_student_runtime.py {{base}} {{output}}

composition-constrained-geometry-unit:
    python3 -m unittest discover -s tools/composition -p test_constrained_geometry.py

composition-unsplash-purpose-witnesses registry output:
    python3 tools/composition/unsplash_purpose_witnesses.py {{registry}} {{output}}

composition-evaluate-unsplash-purposes python study annotations export output:
    {{python}} tools/composition/evaluate_unsplash_purposes.py {{study}} {{annotations}} {{export}} {{output}}

composition-unsplash-purpose-gallery evaluation:
    python3 tools/composition/build_unsplash_purpose_gallery.py {{evaluation}}

composition-unsplash-purpose-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite unsplash-purpose

composition-freeze-unsplash-purposes root export:
    python3 tools/composition/freeze_unsplash_purposes.py {{root}} {{export}}

composition-acquire-florence-regions python output *args:
    {{python}} tools/composition/acquire_florence_regions.py {{output}} {{args}}

composition-probe-florence-regions python assets registry resizer output stage:
    {{python}} tools/composition/probe_florence_regions.py {{assets}} {{registry}} {{resizer}} {{output}} --stage {{stage}}

composition-analyze-florence-regions scout purposes registry output:
    python3 tools/composition/analyze_florence_regions.py {{scout}} {{purposes}} {{registry}} {{output}}

composition-prepare-florence-primary56 admission runtime queries output:
    python3 tools/composition/prepare_florence_primary56.py {{admission}} {{runtime}} {{queries}} {{output}}

composition-unsplash-previews python cache oracle pipeline png_reader output *args:
    {{python}} tools/composition/prepare_unsplash_previews.py {{cache}} {{oracle}} {{pipeline}} {{png_reader}} {{output}} {{args}}

composition-verify-ai-branch capture jxl_root:
    python3 tools/composition/verify_ai_branch.py {{capture}} {{jxl_root}}

composition-build-corpus-png target:
    CARGO_TARGET_DIR={{target}} cargo build --locked --release --manifest-path tools/crop-png/Cargo.toml

composition-acquire-efficientdet-family output:
    python3 tools/composition/acquire_efficientdet_family.py {{output}}

composition-prepare-efficientdet-family root screen:
    python3 tools/composition/prepare_efficientdet_family.py {{root}} {{screen}}

composition-evaluate-efficientdet python manifest output:
    {{python}} tools/runtime-eval/run_models.py {{manifest}} {{output}}

composition-analyze-efficientdet root previous_screen:
    python3 tools/composition/analyze_efficientdet_family.py {{root}} {{previous_screen}}

composition-efficientdet-gallery root output asset_base:
    python3 tools/composition/efficientdet_gallery.py {{root}} {{output}} --asset-base {{asset_base}}

composition-efficientdet-browser root chromium scratch *args:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite efficientdet {{args}}

composition-unsplash-detectors-prepare previews family output:
    python3 tools/composition/unsplash_detectors.py prepare {{previews}} {{family}} {{output}}

composition-unsplash-detectors-analysis root:
    python3 tools/composition/unsplash_detectors.py analyze {{root}}

composition-freeze-unsplash preparation detectors:
    python3 tools/composition/freeze_unsplash_evidence.py {{preparation}} {{detectors}}

composition-efficientdet-profile root python profile_name='profiles':
    bash tools/runtime-eval/efficientdet_profile.sh {{root}} {{python}} {{profile_name}}

composition-efficientdet-perf-summary root:
    python3 tools/composition/summarize_efficientdet_perf.py {{root}}

composition-freeze-efficientdet root:
    python3 tools/composition/freeze_efficientdet_family.py {{root}}

composition-build-corpus-pipeline target manifest:
    CARGO_TARGET_DIR={{target}} cargo build --locked --release --manifest-path {{manifest}} -p zenpipe-cmd

composition-person-roi-gallery root output asset_base:
    python3 tools/composition/person_roi_gallery.py {{root}} {{output}} --asset-base {{asset_base}}

composition-person-roi-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite person-roi

composition-pack-owned-screen screen acquired earlier_galleries output:
    python3 tools/composition/pack_owned_screen.py {{screen}} {{acquired}} {{earlier_galleries}} {{output}}

composition-training-timing-plan cache output python:
    python3 tools/composition/training_timing_plan.py {{cache}} {{output}} --python {{python}}

composition-probe-florence-primary56 python assets registry resizer output plan:
    {{python}} tools/composition/probe_florence_regions.py {{assets}} {{registry}} {{resizer}} {{output}} --stage primary56 --source-plan {{plan}}

composition-cache-florence-extents python root original_cache:
    {{python}} tools/composition/cache_florence_extents.py {{root}} {{original_cache}}

composition-rebind-florence-features python root focal_cache output:
    {{python}} tools/composition/rebind_florence_features.py {{root}} {{focal_cache}} {{output}}

composition-analyze-florence-extents root output:
    python3 tools/composition/analyze_florence_extents.py {{root}} {{output}}

composition-build-florence-extent-gallery root:
    python3 tools/composition/build_florence_extent_gallery.py {{root}}

composition-freeze-florence-evidence scout fits original_cache focal_cache resizer:
    python3 tools/composition/freeze_florence_evidence.py {{scout}} {{fits}} {{original_cache}} {{focal_cache}} {{resizer}}

composition-cache-photo-student-extents python cache auxiliary weights output:
    {{python}} tools/composition/cache_photo_student_extents.py {{cache}} {{auxiliary}} {{weights}} {{output}}

composition-analyze-photo-student-extents root original:
    python3 tools/composition/analyze_photo_student_extents.py {{root}} {{original}}

composition-freeze-photo-extent-transfer root weights:
    python3 tools/composition/freeze_photo_extent_transfer.py {{root}} {{weights}}

composition-prepare-florence-finetune original auxiliary checkpoint output:
    python3 tools/composition/prepare_florence_finetune.py {{original}} {{auxiliary}} {{checkpoint}} {{output}}

composition-fit-florence-finetune python root auxiliary:
    {{python}} tools/composition/fit_extents.py {{root}} --loss l1giou
    {{python}} tools/composition/fit_extent_images.py {{root}} {{auxiliary}} {{root}}/initializers/epoch11.pt

composition-audit-extent-generalization python root auxiliary:
    {{python}} tools/composition/audit_extent_generalization.py {{root}} {{auxiliary}}

composition-analyze-extent-finetune root:
    python3 tools/composition/analyze_extent_finetune.py {{root}}

composition-freeze-extent-finetune root auxiliary:
    python3 tools/composition/freeze_extent_finetune.py {{root}} {{auxiliary}}

composition-extent-augmentation-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_extent_augmentation.py -v

composition-prepare-augmented-extents original auxiliary checkpoint output:
    python3 tools/composition/prepare_florence_finetune.py {{original}} {{auxiliary}} {{checkpoint}} {{output}} --experiment augmented

composition-fit-augmented-extents python root auxiliary:
    {{python}} tools/composition/fit_extent_images.py {{root}} {{auxiliary}} {{root}}/initializers/epoch11.pt --augment flip-color

composition-mirror-packet host local_root remote_root:
    rsync -rt --copy-links --mkpath {{local_root}}/ {{host}}:{{remote_root}}/

composition-extent-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p 'test_composition_extent*.py' -v

composition-audit-augmented-extents python root auxiliary:
    {{python}} tools/composition/audit_extent_generalization.py {{root}} {{auxiliary}} --augmented

composition-analyze-extent-augmentation root baseline:
    python3 tools/composition/analyze_extent_augmentation.py {{root}} {{baseline}}

composition-freeze-extent-augmentation root auxiliary:
    python3 tools/composition/freeze_extent_finetune.py {{root}} {{auxiliary}} --augmented

composition-build-unsplash-content-audit python registry resizer output:
    {{python}} tools/composition/build_unsplash_content_audit.py {{registry}} {{resizer}} {{output}}

composition-finalize-unsplash-content-audit root:
    python3 tools/composition/finalize_unsplash_content_audit.py {{root}}

composition-freeze-unsplash-content-audit root:
    python3 tools/composition/freeze_unsplash_content_audit.py {{root}}

composition-build-owned-content-audit python inputs sources resizer output:
    {{python}} tools/composition/build_unsplash_content_audit.py {{inputs}} {{resizer}} {{output}} --owned-sources {{sources}}

composition-finalize-owned-content-audit root:
    python3 tools/composition/finalize_unsplash_content_audit.py {{root}} --owned-screening

composition-freeze-owned-content-audit root:
    python3 tools/composition/freeze_unsplash_content_audit.py {{root}} --owned-screening

composition-inventory-crop-source-expansion corpus registry output:
    python3 tools/composition/inventory_crop_source_expansion.py {{corpus}} {{registry}} {{output}}

composition-prepare-people-expansion inventory corpus output:
    python3 tools/composition/prepare_people_expansion.py {{inventory}} {{corpus}} {{output}}

composition-people-expansion-oracles python acquired output:
    {{python}} tools/composition/verify_owned_pixels.py {{acquired}} {{output}}/canonical-pixels.json
    {{python}} tools/crop-first/acquisition_oracle.py {{acquired}} {{output}}/png-tensor-oracle.json

composition-screen-supplementary-relationships people owned old_selection synthetic photo_fingerprints binary resizer output:
    python3 tools/composition/screen_supplementary_relationships.py {{people}} {{owned}} {{old_selection}} {{synthetic}} {{photo_fingerprints}} {{binary}} {{resizer}} {{output}}

composition-freeze-supplementary-development people owned owned_review oracles relationships output:
    python3 tools/composition/freeze_supplementary_development.py {{people}} {{owned}} {{owned_review}} {{oracles}} {{relationships}} {{output}}

composition-pack-supplementary-development people review inventory relationships admission binaries output:
    python3 tools/composition/pack_supplementary_development.py {{people}} {{review}} {{inventory}} {{relationships}} {{admission}} {{binaries}} {{output}}

composition-plan-supplementary-extents cohort studies output:
    python3 tools/composition/plan_supplementary_extent_evaluation.py {{cohort}} {{studies}} {{output}}

composition-probe-supplementary-extents python assets registry resizer output plan:
    {{python}} tools/composition/probe_florence_regions.py {{assets}} {{registry}} {{resizer}} {{output}} --stage supplementary15 --source-plan {{plan}}

composition-evaluate-supplementary-extents python cohort root studies models auxiliary:
    {{python}} tools/composition/evaluate_supplementary_extents.py {{cohort}} {{root}} {{studies}} {{models}} {{auxiliary}}

composition-analyze-supplementary-extents root:
    python3 tools/composition/analyze_supplementary_extents.py {{root}}

composition-freeze-supplementary-extents root cohort:
    python3 tools/composition/freeze_supplementary_extents.py {{root}} {{cohort}}

composition-prepare-photo-subjects registry review output:
    python3 tools/composition/prepare_photo_subject_queries.py {{registry}} {{review}} {{output}}

composition-probe-photo-subjects python assets registry resizer output plan:
    {{python}} tools/composition/probe_florence_regions.py {{assets}} {{registry}} {{resizer}} {{output}} --stage photo87 --source-plan {{plan}}

composition-audit-photo-subjects python root cache previews predictions:
    {{python}} tools/composition/audit_photo_subject_references.py {{root}} {{cache}} {{previews}} {{predictions}}

composition-photo-subject-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite photo-subjects

composition-prepare-photo-people registry review output:
    python3 tools/composition/prepare_photo_subject_queries.py {{registry}} {{review}} {{output}} --people-only

composition-probe-photo-people python assets registry resizer output plan:
    {{python}} tools/composition/probe_florence_regions.py {{assets}} {{registry}} {{resizer}} {{output}} --stage photo-people20 --source-plan {{plan}}

composition-audit-photo-generic-subjects python root cache previews predictions:
    {{python}} tools/composition/audit_photo_subject_references.py {{root}} {{cache}} {{previews}} {{predictions}} --generic-people

composition-freeze-photo-subjects root:
    python3 tools/composition/freeze_photo_subject_references.py {{root}}

composition-cache-photo-retention python rank_cache regions weights protocol predictions output:
    {{python}} tools/composition/cache_photo_retention.py {{rank_cache}} {{regions}} {{weights}} {{protocol}} {{predictions}} {{output}}

composition-fit-photo-retention python root:
    {{python}} tools/composition/fit_photo_retention.py {{root}}

composition-audit-photo-retention python root:
    {{python}} tools/composition/audit_photo_retention_heads.py {{root}}

composition-freeze-photo-retention root:
    python3 tools/composition/freeze_photo_retention_heads.py {{root}}

composition-plan-retention-supplementary heads cohort references output:
    python3 tools/composition/plan_retention_supplementary.py {{heads}} {{cohort}} {{references}} {{output}}

composition-evaluate-retention-supplementary python root references weights:
    {{python}} tools/composition/evaluate_retention_supplementary.py {{root}} {{references}} {{weights}}

composition-build-retention-review python root:
    {{python}} tools/composition/build_retention_review.py {{root}}

composition-retention-review-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite retention-review

composition-freeze-retention-supplementary root:
    python3 tools/composition/freeze_retention_supplementary.py {{root}}

composition-retention-loss-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_retention_loss.py -v

composition-prepare-retention-loss baseline output:
    python3 tools/composition/prepare_retention_loss_control.py {{baseline}} {{output}}

composition-fit-natural-photo-retention python root:
    {{python}} tools/composition/fit_photo_retention.py {{root}} --class-balance natural

composition-compare-retention-losses natural balanced:
    python3 tools/composition/compare_retention_losses.py {{natural}} {{balanced}}

composition-freeze-natural-retention root:
    python3 tools/composition/freeze_photo_retention_heads.py {{root}} --natural-loss

composition-plan-natural-retention-supplementary heads cohort references output digest:
    python3 tools/composition/plan_retention_supplementary.py {{heads}} {{cohort}} {{references}} {{output}} --heads-sha256 {{digest}}

composition-build-natural-retention-review python root:
    {{python}} tools/composition/build_retention_review.py {{root}} --natural-loss

composition-compare-supplementary-retention-losses natural balanced:
    python3 tools/composition/compare_retention_losses.py {{natural}} {{balanced}} --supplementary

composition-freeze-natural-retention-supplementary root digest:
    python3 tools/composition/freeze_retention_supplementary.py {{root}} --natural-loss --plan-sha256 {{digest}}

composition-acquire-miap-metadata root:
    python3 tools/composition/acquire_miap_metadata.py {{root}}

composition-freeze-miap-metadata root:
    python3 tools/composition/freeze_miap_metadata.py {{root}}

composition-join-miap-image-information metadata output:
    python3 tools/composition/join_miap_image_information.py {{metadata}} {{output}}

composition-freeze-miap-image-information root:
    python3 tools/composition/freeze_miap_image_information.py {{root}}

composition-miap-source-unit:
    python3 -m unittest discover -s tools/composition -p test_miap_source_groups.py -v

composition-plan-miap-pilot annotations information corpus photographic supplementary canonical output:
    python3 tools/composition/plan_miap_people_pilot.py {{annotations}} {{information}} {{corpus}} {{photographic}} {{supplementary}} {{canonical}} {{output}}

composition-freeze-miap-pilot root:
    python3 tools/composition/freeze_miap_pilot.py {{root}}

composition-acquire-miap-witnesses python plan annotations pipeline png_reader output *args:
    {{python}} tools/composition/acquire_miap_witnesses.py {{plan}} {{annotations}} {{pipeline}} {{png_reader}} {{output}} {{args}}

composition-miap-witness-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite miap-witness

composition-review-miap-witnesses root binary photographic supplementary:
    python3 tools/composition/review_miap_witnesses.py {{root}} {{binary}} {{photographic}} {{supplementary}}

composition-freeze-miap-witnesses root:
    python3 tools/composition/freeze_miap_witnesses.py {{root}}

composition-plan-miap-group-pilot parent witnesses annotations output:
    python3 tools/composition/plan_miap_group_pilot.py {{parent}} {{witnesses}} {{annotations}} {{output}}

composition-freeze-miap-group-pilot root parent:
    python3 tools/composition/freeze_miap_group_pilot.py {{root}} {{parent}}

composition-mirror-goal-progress host:
    rsync -rt --mkpath /mnt/v/output/zensally/composition-goal-progress*.json {{host}}:/mnt/tower/output/zensally/

composition-miap-preview-parity python witnesses pipeline reader output:
    {{python}} tools/composition/check_miap_preview_parity.py {{witnesses}} {{pipeline}} {{reader}} {{output}}

composition-acquire-miap-pilot python plan annotations witnesses pipeline reader parity output *args:
    {{python}} tools/composition/acquire_miap_pilot.py {{plan}} {{annotations}} {{witnesses}} {{pipeline}} {{reader}} {{parity}} {{output}} {{args}}

composition-miap-gray-readback python png rgb output:
    {{python}} -c 'import hashlib,json,sys; from pathlib import Path; from PIL import Image; p,r,o=map(Path,sys.argv[1:]); im=Image.open(p).convert("RGB"); raw=r.read_bytes(); assert im.tobytes()==raw; assert len(raw)==im.width*im.height*3; assert all(raw[i]==raw[i+1]==raw[i+2] for i in range(0,len(raw),3)); result=dict(size=list(im.size),rgb_sha256=hashlib.sha256(raw).hexdigest(),independent_rgb_exact=True,equal_gray_channels=True); o.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result))' {{png}} {{rgb}} {{output}}

composition-caption-miap-content python assets sources resizer output stage:
    {{python}} tools/composition/caption_miap_content.py {{assets}} {{sources}} {{resizer}} {{output}} --stage {{stage}}

composition-screen-miap-relationships acquired binary photographic supplementary output:
    python3 tools/composition/screen_miap_relationships.py {{acquired}} {{binary}} {{photographic}} {{supplementary}} {{output}}

composition-build-miap-content-review acquired captions relationships output:
    python3 tools/composition/build_miap_content_review.py {{acquired}} {{captions}} {{relationships}} {{output}}

composition-miap-content-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite miap-content

composition-prepare-miap-detectors acquired family output:
    python3 tools/composition/prepare_miap_detector_screen.py {{acquired}} {{family}} {{output}}

composition-finalize-miap-content root:
    python3 tools/composition/finalize_miap_content_review.py {{root}}

composition-miap-detector-matching-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_miap_detector_matching.py -v

composition-score-miap-detectors python root acquired content output:
    {{python}} tools/composition/score_miap_detectors.py {{root}} {{acquired}} {{content}} {{output}}

composition-freeze-miap-screen action:
    python3 tools/composition/freeze_miap_screen.py {{action}}

composition-miap-legacy-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite unsplash-content

composition-cache-miap-transfer python acquired review resizer output:
    {{python}} tools/composition/cache_miap_transfer.py {{acquired}} {{review}} {{resizer}} {{output}}

composition-person-relationship-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_person_relationships.py -v

composition-fit-miap-transfer python rank auxiliary weights output:
    {{python}} tools/composition/fit_miap_transfer.py {{rank}} {{auxiliary}} {{weights}} {{output}}

composition-analyze-miap-transfer fits output:
    python3 tools/composition/analyze_miap_transfer.py {{fits}} {{output}}

composition-freeze-miap-transfer root:
    python3 tools/composition/freeze_miap_transfer.py {{root}}

composition-cache-miap-ranks python auxiliary acquired models split output:
    {{python}} tools/composition/cache_miap_ranks.py {{auxiliary}} {{acquired}} {{models}} {{split}} {{output}}

composition-record-miap-progress run producer:
    python3 tools/composition/record_miap_progress.py {{run}} {{producer}}

composition-fit-miap-broad-ranks python old_cache new_cache weights output:
    {{python}} tools/composition/fit_miap_broad_ranks.py {{old_cache}} {{new_cache}} {{weights}} {{output}}

composition-analyze-miap-broad-ranks fits output:
    python3 tools/composition/analyze_miap_broad_ranks.py {{fits}} {{output}}

composition-plan-miap-rank-review acquired review output:
    python3 tools/composition/plan_miap_rank_review.py {{acquired}} {{review}} {{output}}

composition-miap-rank-gallery python cache fits analysis plan output:
    {{python}} tools/composition/miap_rank_gallery.py {{cache}} {{fits}} {{analysis}} {{plan}} {{output}}

composition-miap-rank-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite miap-rank

composition-api-models host env_file output:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/api_model_inventory.py {{output}}

composition-subject-policy-unit:
    python3 -m unittest discover -s tools/composition -p test_subject_policy.py -v

crop-additional-budget-unit:
    python3 -m unittest discover -s tools/crop-first -p test_additional_budget.py -v

crop-subject-pilot-plan output:
    python3 tools/crop-first/sunburst_subject_pilot.py {{output}}

crop-subject-pilot-generate host env_file plan cache output authorization:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/sunburst_generate.py {{plan}} {{cache}} {{output}} --budget-ceiling-usd 12 --sized-reservations --authorization {{authorization}}

composition-freeze-miap-broad-ranks root:
    python3 tools/composition/freeze_miap_broad_ranks.py {{root}}

crop-api-budget-unit:
    python3 -m unittest discover -s tools/crop-first -p test_api_budget.py -v

crop-subject-zones host env_file task cache budget authorization output *args:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/subject_zones.py {{task}} {{cache}} {{budget}} {{authorization}} {{output}} {{args}}

crop-prepare-subject-tasks python source resizer reader writer output:
    {{python}} tools/crop-first/prepare_subject_tasks.py {{source}} {{resizer}} {{reader}} {{writer}} {{output}}

crop-sunburst-unit:
    python3 -m unittest discover -s tools/crop-first -p test_sunburst.py -v

crop-subject-zones-unit:
    python3 -m unittest discover -s tools/crop-first -p test_subject_zones.py -v

crop-repair-subject-aliases entry task aliases output:
    python3 tools/crop-first/repair_subject_aliases.py {{entry}} {{task}} {{aliases}} {{output}}

crop-subject-alias-unit:
    python3 -m unittest discover -s tools/crop-first -p test_subject_aliases.py -v

# Geometry validation never admits unreviewed teacher references.
crop-subject-geometry zones width height output:
    python3 tools/crop-first/subject_geometry.py {{zones}} {{width}} {{height}} {{output}}

crop-subject-geometry-unit:
    python3 -m unittest discover -s tools/crop-first -p test_subject_geometry.py -v

crop-render-subject-geometry python audit sources renderer resizer output:
    {{python}} tools/crop-first/render_subject_geometry.py {{audit}} {{sources}} {{renderer}} {{resizer}} {{output}}

crop-subject-expansion-plan output:
    python3 tools/crop-first/sunburst_subject_expansion.py {{output}}

crop-subject-expansion-generate host env_file plan cache output authorization carry:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/sunburst_generate.py {{plan}} {{cache}} {{output}} --budget-ceiling-usd 12 --sized-reservations --authorization {{authorization}} --carry-journal {{carry}}

crop-consult-architecture host env_file cache budget authorization output:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/consult_crop_architecture.py {{cache}} {{budget}} {{authorization}} {{output}}

crop-subject-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite subject-choice

crop-record-subject-progress root progress *args:
    python3 tools/crop-first/record_subject_progress.py {{root}} {{progress}} {{args}}

crop-freeze-subject-pilot run output:
    python3 tools/crop-first/freeze_subject_pilot.py {{run}} {{output}}

crop-plan-real-subjects acquired review output:
    python3 tools/crop-first/plan_real_subject_tasks.py {{acquired}} {{review}} {{output}}

crop-batch-subject-zones host env_file tasks cache budget authorization output:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/batch_subject_zones.py {{tasks}} {{cache}} {{budget}} {{authorization}} {{output}} --workers 3

composition-acceptable-sets-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_acceptable_crop_sets.py -v

crop-cache-subject-sets python tasks responses rank_cache output:
    {{python}} tools/crop-first/cache_subject_sets.py {{tasks}} {{responses}} {{rank_cache}} {{output}}

composition-fit-subject-sets python old_cache set_cache weights output:
    {{python}} tools/composition/fit_subject_sets.py {{old_cache}} {{set_cache}} {{weights}} {{output}}

crop-funded-gallery root authorization:
    python3 tools/crop-first/sunburst_gallery.py {{root}} --authorization {{authorization}}

crop-funded-native-verify python root cache output carry authorization:
    {{python}} tools/crop-first/sunburst_verify.py {{root}} {{cache}} {{output}} --carry-journal {{carry}} --authorization {{authorization}}

crop-funded-source-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite sunburst-v4

crop-verify-subject-responses cache budget output:
    python3 tools/crop-first/verify_subject_responses.py {{cache}} {{budget}} {{output}}

crop-prepare-funded-tasks python source resizer reader writer output policy:
    {{python}} tools/crop-first/prepare_subject_tasks.py {{source}} {{resizer}} {{reader}} {{writer}} {{output}} --policy {{policy}} --storage-budget-mib 384

composition-fit-subject-replay-control python old_cache set_cache weights output:
    {{python}} tools/composition/fit_subject_sets.py {{old_cache}} {{set_cache}} {{weights}} {{output}} --arms frozen-image --new-loss-weight 0

composition-analyze-subject-sets python fits replay_control cache output:
    {{python}} tools/composition/analyze_subject_sets.py {{fits}} {{replay_control}} {{cache}} {{output}}

crop-freeze-subject-expansion run output:
    python3 tools/crop-first/freeze_subject_expansion.py {{run}} {{output}}

crop-verify-dataset-blobs repository revision expected_images output *args:
    python3 tools/crop-first/verify_dataset_blobs.py {{repository}} {{revision}} {{expected_images}} {{output}} {{args}}

crop-compare-subject-teachers run output:
    python3 tools/crop-first/compare_subject_teachers.py {{run}} {{output}}

composition-subject-sets-gallery python run rank_cache initializer output *args:
    {{python}} tools/composition/subject_sets_gallery.py {{run}} {{rank_cache}} {{initializer}} {{output}} {{args}}

composition-subject-bank-unit:
    python3 -m unittest discover -s tools/composition -p test_subject_candidate_bank.py -v

crop-plan-sol-primary run output:
    python3 tools/crop-first/plan_sol_primary.py {{run}} {{output}}

composition-audit-subject-bank-density python run rank_cache output:
    {{python}} tools/composition/audit_subject_bank_density.py {{run}} {{rank_cache}} {{output}}

crop-freeze-subject-fits run output:
    python3 tools/crop-first/freeze_subject_fits.py {{run}} {{output}}

crop-plan-subject-replacements source output:
    python3 tools/crop-first/plan_subject_replacements.py {{source}} {{output}}

crop-freeze-subject-replacements run output:
    python3 tools/crop-first/freeze_subject_replacements.py {{run}} {{output}}

crop-freeze-subject-followup run bank output:
    python3 tools/crop-first/freeze_subject_followup.py {{run}} {{bank}} {{output}}

crop-derive-real-sol-view tasks responses output:
    python3 tools/crop-first/derive_subject_response_view.py {{tasks}} {{responses}} {{output}} --source-prefix openimages: --expected 72

crop-subject-response-view-unit:
    python3 -m unittest discover -s tools/crop-first -p test_subject_response_view.py -v

crop-verify-existing-real-sol-view tasks responses output:
    python3 tools/crop-first/derive_subject_response_view.py {{tasks}} {{responses}} {{output}} --source-prefix openimages: --expected 72 --verify-existing

crop-dataset-source-review-unit:
    python3 -m unittest discover -s tools/crop-first -p test_dataset_source_review.py -v

composition-audit-subject-cross-teacher python run rank_cache initializer output:
    {{python}} tools/composition/audit_subject_cross_teacher.py {{run}} {{rank_cache}} {{initializer}} {{output}}

composition-cache-subject-bank python run reference_cache rank_cache output bank:
    {{python}} tools/composition/cache_subject_bank_control.py {{run}} {{reference_cache}} {{rank_cache}} {{output}} --bank {{bank}}

composition-subject-analysis-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_subject_analysis_baseline.py -v

composition-freeze-subject-bank root:
    python3 tools/composition/freeze_subject_bank_control.py {{root}}

composition-audit-subject-bank-transfer python run fits transfer initializer output:
    {{python}} tools/composition/audit_subject_bank_transfer.py {{run}} {{fits}} {{transfer}} {{initializer}} {{output}}

composition-direct-crop-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_direct_crop.py -v

composition-cache-direct-crop python run references initializer output:
    {{python}} tools/composition/cache_direct_crop.py {{run}} {{references}} {{initializer}} {{output}}

composition-fit-direct-crop python cache output:
    {{python}} tools/composition/fit_direct_crop.py {{cache}} {{output}}

composition-analyze-direct-crop python root output:
    {{python}} tools/composition/analyze_direct_crop.py {{root}} {{output}}

composition-freeze-subject-successor kind root:
    python3 tools/composition/freeze_subject_successors.py {{kind}} {{root}}

composition-prepare-subject-face-cues tasks models output:
    python3 tools/composition/prepare_subject_face_cues.py {{tasks}} {{models}} {{output}}

composition-evaluate-subject-face-cues python root:
    {{python}} tools/runtime-eval/run_models.py {{root}}/manifest.json {{root}}/local

composition-audit-subject-face-cues run root:
    python3 tools/composition/audit_subject_face_cues.py {{run}} {{root}}

composition-audit-local-crop-guards python run cache fits cues output:
    {{python}} tools/composition/audit_local_crop_guards.py {{run}} {{cache}} {{fits}} {{cues}} {{output}}

composition-direct-crop-gallery python run direct rank_gallery output:
    {{python}} tools/composition/direct_crop_gallery.py {{run}} {{direct}} {{rank_gallery}} {{output}}

composition-region-context-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_region_context.py -v

composition-cache-region-context python direct_cache cues output:
    {{python}} tools/composition/cache_region_context.py {{direct_cache}} {{cues}} {{output}}

composition-checkpoint-snapshot-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_checkpoint_snapshot.py -v

composition-fit-region-context python cache regions variant output:
    {{python}} tools/composition/fit_direct_crop.py {{cache}} {{output}} --region-cache {{regions}} --region-variant {{variant}} --compact-states

composition-subject-origin-unit:
    python3 -m unittest discover -s tools/composition -p test_subject_origin_projection.py -v

composition-analyze-region-context python root direct output:
    {{python}} tools/composition/analyze_region_context.py {{root}} {{direct}} {{output}}

composition-audit-origin-projection python run direct regions cues output *args:
    {{python}} tools/composition/audit_subject_origin_projection.py {{run}} {{direct}} {{regions}} {{cues}} {{output}} {{args}}

composition-prepare-subject-native-faces cues run binary output:
    python3 tools/composition/subject_native_faces.py prepare {{cues}} {{run}} {{binary}} {{output}}

composition-evaluate-subject-native-faces root binary:
    python3 tools/runtime-eval/run_models.py {{root}}/manifest.json {{root}}/local --rust {{binary}} --models ultraface yunet

composition-audit-subject-native-faces cues run binary output:
    python3 tools/composition/subject_native_faces.py audit {{cues}} {{run}} {{binary}} {{output}}

composition-origin-projection-gallery python direct regions projection cues rank_gallery output:
    {{python}} tools/composition/origin_projection_gallery.py {{direct}} {{regions}} {{projection}} {{cues}} {{rank_gallery}} {{output}}

composition-freeze-region-evidence kind root:
    python3 tools/composition/freeze_region_evidence.py {{kind}} {{root}}

composition-subject-tiled-faces-unit:
    python3 -m unittest discover -s tools/composition -p test_subject_tiled_faces.py -v

composition-subject-tiled-faces action native run binary output:
    python3 tools/composition/subject_tiled_faces.py {{action}} {{native}} {{run}} {{binary}} {{output}}

composition-evaluate-subject-tiled-faces root binary:
    python3 tools/runtime-eval/run_models.py {{root}}/manifest.json {{root}}/local --rust {{binary}} --models yunet

composition-prepare-real-attention native tiles packet teachers resizer output:
    python3 tools/composition/prepare_real_attention.py {{native}} {{tiles}} {{packet}} {{teachers}} {{resizer}} {{output}}

composition-attention-heads-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_attention_heads.py -v

composition-cache-real-attention python root set_cache direct_cache cues weights:
    {{python}} tools/composition/cache_real_attention.py {{root}} {{set_cache}} {{direct_cache}} {{cues}} {{weights}}

composition-attention-scalar-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_attention_scalar_head.py -v

crop-label-pilot-budget-unit:
    python3 -m unittest discover -s tools/crop-first -p test_label_pilot_budget.py -v

crop-prepare-label-quality python root study attention native resizer encoder:
    {{python}} tools/crop-first/prepare_label_quality.py {{root}} {{study}} {{attention}} {{native}} {{resizer}} {{encoder}}

crop-label-quality host env_file task root output *args:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/label_quality.py {{task}} {{root}} {{output}} {{args}}

crop-corridor-unit:
    python3 -m unittest discover -s tools/crop-first -p test_crop_corridors.py -v

composition-audit-attention-supervision python cache output:
    {{python}} tools/composition/audit_attention_supervision.py {{cache}} {{output}}

composition-fit-attention-scalar python cache output *args:
    {{python}} tools/composition/fit_attention_scalar.py {{cache}} {{output}} {{args}}

composition-analyze-attention-scalar python root:
    {{python}} tools/composition/analyze_attention_scalar.py {{root}}

composition-evaluate-attention-fallback python fits cache weights output *args:
    {{python}} tools/composition/evaluate_attention_fallback.py {{fits}} {{cache}} {{weights}} {{output}} {{args}}

composition-compare-attention-controls python fits output *args:
    {{python}} tools/composition/compare_attention_controls.py {{fits}} {{output}} {{args}}

composition-distill-attention-linear python cache teachers output *args:
    {{python}} tools/composition/distill_attention_linear.py {{cache}} {{teachers}} {{output}} {{args}}

composition-cache-old-attention-replay python cache weights output:
    {{python}} tools/composition/cache_old_attention_replay.py {{cache}} {{weights}} {{output}}

composition-fit-relationship-rehearsal python old new relationships teachers output:
    {{python}} tools/composition/fit_relationship_rehearsal.py {{old}} {{new}} {{relationships}} {{teachers}} {{output}}

composition-analyze-relationship-rehearsal python fits gates output:
    {{python}} tools/composition/analyze_relationship_rehearsal.py {{fits}} {{gates}} {{output}}

composition-relationship-rehearsal-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_relationship_rehearsal.py -v

composition-export-residual-native python cache fits output:
    {{python}} tools/composition/export_residual_native.py {{cache}} {{fits}} {{output}}

composition-residual-export-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_residual_export.py -v

composition-build-residual-native model baker target:
    ZENSALLY_RESIDUAL_MODEL={{model}} CARGO_TARGET_DIR={{target}} cargo build --offline --config 'patch.crates-io.zenpredict-bake.path="{{baker}}"' --release --manifest-path tools/residual-native/Cargo.toml

composition-clippy-residual-native model baker target:
    ZENSALLY_RESIDUAL_MODEL={{model}} CARGO_TARGET_DIR={{target}} cargo clippy --offline --config 'patch.crates-io.zenpredict-bake.path="{{baker}}"' --release --manifest-path tools/residual-native/Cargo.toml -- -D warnings

composition-qualify-residual-native python root cache binary:
    {{python}} tools/composition/qualify_residual_native.py {{root}} {{cache}} {{binary}}

composition-residual-runtime-plan root:
    python3 tools/composition/residual_runtime_plan.py {{root}}

composition-profile-residual-runtime root:
    test ! -e {{root}}/residual.perf-stat
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -e cycles,instructions,cache-misses -o {{root}}/residual.perf-stat -- /usr/bin/time -v -o {{root}}/residual.time taskset -c 2 {{root}}/binaries/residual-head bench {{root}}/hidden.bin residual 22 20000 1056

composition-heap-residual-runtime root name repeats:
    mkdir {{root}}/heap-{{name}}
    heaptrack -o {{root}}/heap-{{name}}/trace {{root}}/binaries/residual-head bench {{root}}/hidden.bin residual 22 {{repeats}} 1056

composition-record-residual-runtime root:
    test ! -e {{root}}/calls.perf
    sudo perf record --no-buildid-cache --no-buildid -F 199 --call-graph dwarf,4096 -o {{root}}/calls.perf -- taskset -c 2 {{root}}/binaries/residual-head bench {{root}}/hidden.bin residual 22 4000 1056
    sudo perf report -i {{root}}/calls.perf --stdio --no-children --percent-limit 1 > {{root}}/calls.txt
    sudo chown "$(id -u):$(id -g)" {{root}}/calls.perf

composition-print-residual-heap root name:
    test ! -e {{root}}/heap-{{name}}/report.txt
    heaptrack_print -f {{root}}/heap-{{name}}/trace.zst > {{root}}/heap-{{name}}/report.txt

composition-summarize-residual-runtime root target:
    python3 tools/composition/summarize_residual_runtime.py {{root}} {{target}}

composition-export-rehearsal-network python old new weights student output *args:
    {{python}} tools/composition/export_rehearsal_network.py {{old}} {{new}} {{weights}} {{student}} {{output}} {{args}}

composition-rehearsal-network-plan python old new export *args:
    python3 tools/composition/rehearsal_network_plan.py {{python}} {{old}} {{new}} {{export}} {{args}}

composition-verify-rehearsal-bindings python old new export:
    {{python}} tools/composition/verify_rehearsal_bindings.py {{old}} {{new}} {{export}}

composition-rehearsal-gather-guard-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_rehearsal_gather_guard.py -v

composition-verify-rehearsal-gather python raw gather:
    {{python}} tools/composition/verify_rehearsal_gather.py {{raw}} {{gather}}

composition-summarize-rehearsal-network python raw gather:
    {{python}} tools/composition/summarize_rehearsal_network.py {{raw}} {{gather}}

composition-profile-rehearsal-network python old new export:
    test ! -e {{export}}/network.perf-stat
    sudo perf stat --cputype=core -M tma_memory_bound,tma_core_bound -e cycles,instructions,cache-misses -o {{export}}/network.perf-stat -- /usr/bin/time -v -o {{export}}/network.time sudo -u "$(id -un)" env TMPDIR="$HOME/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 {{python}} tools/composition/bench_rehearsal_network.py {{old}} {{new}} {{export}} --arm candidate --phase bank --repeats 1

composition-audit-attention-fusion python fits cache output:
    {{python}} tools/composition/audit_attention_fusion.py {{fits}} {{cache}} {{output}}

composition-attention-fusion-unit python:
    {{python}} -m unittest discover -s tools/composition -p test_attention_fusion.py -v

composition-relationship-unit python:
    PYTHONPATH=training {{python}} -m unittest discover -s training -p test_relationship_features.py -v

composition-cache-relationships python cache attention output:
    {{python}} tools/composition/cache_relationship_features.py {{cache}} {{attention}} {{output}}

composition-render-relationship-witnesses python cache control native links resizer encoder output *args:
    {{python}} tools/composition/render_relationship_witnesses.py {{cache}} {{control}} {{native}} {{links}} {{resizer}} {{encoder}} {{output}} {{args}}

composition-check-relationship-gallery python root url *args:
    {{python}} tools/composition/check_relationship_gallery.py {{root}} {{url}} {{args}}

composition-verify-rehearsal-witness-outputs python gallery cache weights control serving:
    {{python}} tools/composition/verify_rehearsal_witness_outputs.py {{gallery}} {{cache}} {{weights}} {{control}} {{serving}}

composition-audit-rehearsal-faces python cache control detections output *args:
    {{python}} tools/composition/audit_rehearsal_face_candidates.py {{cache}} {{control}} {{detections}} {{output}} {{args}}

composition-render-face-candidate-witnesses python audit native resizer encoder *args:
    {{python}} tools/composition/render_face_candidate_witnesses.py {{audit}} {{native}} {{resizer}} {{encoder}} {{args}}

composition-diagnose-face-candidate-misses python cache direct audit detections *args:
    {{python}} tools/composition/diagnose_face_candidate_misses.py {{cache}} {{direct}} {{audit}} {{detections}} {{args}}

composition-audit-pose-constraint-framing python cache audit pose:
    {{python}} tools/composition/audit_pose_constraint_framing.py {{cache}} {{audit}} {{pose}}

composition-verify-guarded-rehearsal-network python cache audit serving detections:
    {{python}} tools/composition/verify_guarded_rehearsal_network.py {{cache}} {{audit}} {{serving}} {{detections}}

composition-prepare-old-guard-panel python cache face_manifest binary output:
    {{python}} tools/composition/prepare_old_guard_panel.py {{cache}} {{face_manifest}} {{binary}} {{output}}

composition-evaluate-old-guard-native root binary:
    python3 tools/runtime-eval/run_models.py {{root}}/manifest.json {{root}}/native --rust {{binary}} --models ultraface yunet
    python3 tools/runtime-eval/run_models.py {{root}}/tiles/manifest.json {{root}}/tiles/native --rust {{binary}} --models yunet

composition-evaluate-old-guard-mediapipe python root:
    {{python}} tools/runtime-eval/run_models.py {{root}}/manifest.json {{root}}/mediapipe

composition-collect-old-guard-regions root:
    python3 tools/composition/collect_old_guard_regions.py {{root}}

composition-evaluate-old-guard-quality python root cache control serving:
    {{python}} tools/composition/evaluate_old_guard_quality.py {{root}} {{cache}} {{control}} {{serving}}

composition-render-old-guard-witnesses python root resizer encoder:
    {{python}} tools/composition/render_old_guard_witnesses.py {{root}} {{resizer}} {{encoder}}

composition-verify-tile-face-mapping root output:
    python3 tools/composition/verify_tile_face_mapping.py {{root}} {{output}}

composition-audit-face-corroboration python faces face_manifest direct original miap_objects unsplash_objects output *args:
    {{python}} tools/composition/audit_face_corroboration.py {{faces}} {{face_manifest}} {{direct}} {{original}} {{miap_objects}} {{unsplash_objects}} {{output}} {{args}}

composition-render-corroboration-witnesses python audit manifest:
    {{python}} tools/composition/render_corroboration_witnesses.py {{audit}} {{manifest}}

composition-prepare-fit-pair-review python cache direct control resizer encoder output *args:
    {{python}} tools/composition/prepare_fit_pair_review.py {{cache}} {{direct}} {{control}} {{resizer}} {{encoder}} {{output}} {{args}}

composition-fit-visual-preferences python old new control review output:
    {{python}} tools/composition/fit_visual_preferences.py {{old}} {{new}} {{control}} {{review}} {{output}}

composition-evaluate-visual-preference-quality python fits cache weights output:
    {{python}} tools/composition/evaluate_visual_preference_quality.py {{fits}} {{cache}} {{weights}} {{output}}

composition-probe-visual-preference-capacity python old new control review fits output:
    {{python}} tools/composition/probe_visual_preference_capacity.py {{old}} {{new}} {{control}} {{review}} {{fits}} {{output}}

composition-verify-fit-pair-review python root:
    {{python}} tools/composition/verify_fit_pair_review.py {{root}}

composition-nested-selection-unit:
    python3 -m unittest discover -s tools/composition -p test_nested_pair_selection.py -v

composition-audit-nested-selection python old_hidden old_rank new control output *args:
    {{python}} tools/composition/audit_nested_selection.py {{old_hidden}} {{old_rank}} {{new}} {{control}} {{output}} {{args}}

composition-verify-nested-selection-network python old new audit serving:
    {{python}} tools/composition/verify_nested_selection_network.py {{old}} {{new}} {{audit}} {{serving}}

composition-render-nested-selection python audit review resizer encoder:
    {{python}} tools/composition/render_nested_selection.py {{audit}} {{review}} {{resizer}} {{encoder}}

composition-audit-nested-oracle python cache output:
    {{python}} tools/composition/audit_nested_oracle.py {{cache}} {{output}}

composition-pair-training-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_nested_pair_training.py -v

composition-fit-nested-pairs python old rank new control output:
    {{python}} tools/composition/fit_nested_pairs.py {{old}} {{rank}} {{new}} {{control}} {{output}}

composition-evaluate-nested-pairs python fits hidden cache new weights output:
    {{python}} tools/composition/evaluate_nested_pairs.py {{fits}} {{hidden}} {{cache}} {{new}} {{weights}} {{output}}

composition-probe-nested-pair-capacity python hidden rank initializer output:
    {{python}} tools/composition/probe_nested_pair_capacity.py {{hidden}} {{rank}} {{initializer}} {{output}}

composition-pair-relationship-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_pair_relationship_head.py -v

composition-fit-pair-relationship python old rank new initializer output:
    {{python}} tools/composition/fit_pair_relationship.py {{old}} {{rank}} {{new}} {{initializer}} {{output}}

composition-qualify-pair-relationship python fits old new new_hidden weights endpoint output:
    {{python}} tools/composition/qualify_pair_relationship.py {{fits}} {{old}} {{new}} {{new_hidden}} {{weights}} {{endpoint}} {{output}}

composition-render-pair-relationship python fits old new metadata resizer encoder:
    {{python}} tools/composition/render_pair_relationship.py {{fits}} {{old}} {{new}} {{metadata}} {{resizer}} {{encoder}}

composition-audit-face-label-coverage python cache direct endpoint pairs output:
    {{python}} tools/composition/audit_face_label_coverage.py {{cache}} {{direct}} {{endpoint}} {{pairs}} {{output}}

composition-candidate-cut-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_candidate_cut_head.py -v

composition-fit-candidate-cut python cache output:
    {{python}} tools/composition/fit_candidate_cut.py {{cache}} {{output}}

composition-candidate-cut-routing-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_candidate_cut_routing.py -v

composition-evaluate-candidate-cut-routing python fits cache old_hidden old_rank endpoint pairs coverage output *args:
    {{python}} tools/composition/evaluate_candidate_cut_routing.py {{fits}} {{cache}} {{old_hidden}} {{old_rank}} {{endpoint}} {{pairs}} {{coverage}} {{output}} {{args}}

composition-cache-old-attention python root guard rank:
    {{python}} tools/composition/cache_old_attention_features.py {{root}} {{guard}} {{rank}}

composition-qualify-candidate-cut python fits new old rank auxiliary endpoint pairs output:
    {{python}} tools/composition/qualify_candidate_cut.py {{fits}} {{new}} {{old}} {{rank}} {{auxiliary}} {{endpoint}} {{pairs}} {{output}}

composition-analyze-candidate-cut python fits cache output:
    {{python}} tools/composition/analyze_candidate_cut.py {{fits}} {{cache}} {{output}}

composition-render-candidate-cut python fits cache metadata resizer encoder:
    {{python}} tools/composition/render_candidate_cut.py {{fits}} {{cache}} {{metadata}} {{resizer}} {{encoder}}

composition-person-risk-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_person_risk_policy.py -v

composition-evaluate-person-risk python fits miap_objects unsplash_objects output:
    {{python}} tools/composition/evaluate_person_risk.py {{fits}} {{miap_objects}} {{unsplash_objects}} {{output}}

composition-qualify-person-risk python fits study miap_objects unsplash_objects output:
    {{python}} tools/composition/qualify_person_risk.py {{fits}} {{study}} {{miap_objects}} {{unsplash_objects}} {{output}}

composition-render-person-risk python fits study miap_objects unsplash_objects resizer encoder:
    {{python}} tools/composition/render_person_risk.py {{fits}} {{study}} {{miap_objects}} {{unsplash_objects}} {{resizer}} {{encoder}}

composition-analyze-person-risk python study cache coverage output:
    {{python}} tools/composition/analyze_person_risk.py {{study}} {{cache}} {{coverage}} {{output}}

composition-nested-origin-unit python:
    PYTHONPATH=tools/composition {{python}} -m unittest discover -s tools/composition -p test_refine_nested_origin.py -v

composition-qualify-offbank-teacher python rank packet output:
    {{python}} tools/composition/qualify_offbank_teacher.py {{rank}} {{packet}} {{output}}

composition-prepare-old-attention guard attention resizer output:
    python3 tools/composition/prepare_old_attention.py {{guard}} {{attention}} {{resizer}} {{output}}

composition-run-attention python kind root output teachers *args:
    {{python}} tools/composition/attention.py {{kind}} {{root}} {{output}} --teachers {{teachers}} {{args}}

crop-corridor-proposal-unit:
    python3 -m unittest discover -s tools/crop-first -p test_corridor_proposals.py -v

crop-label-quality-unit:
    python3 -m unittest discover -s tools/crop-first -p test_label_quality.py -v

crop-propose-corridors host env_file task root output *args:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/propose_crop_corridors.py {{task}} {{root}} {{output}} {{args}}

crop-render-corridors python root case_id result native resizer encoder output *args:
    {{python}} tools/crop-first/render_crop_corridors.py {{root}} {{case_id}} {{result}} {{native}} {{resizer}} {{encoder}} {{output}} {{args}}

crop-batch-corridors host env_file tasks root output:
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/batch_subject_zones.py {{tasks}} {{root}}/vision-cache {{root}}/testing {{root}}/authorization.json {{output}} --workers 2 --protocol crop-corridors

crop-batch-label-quality host env_file tasks root output workers="1":
    python3 tools/crop-first/with_api_key.py {{host}} {{env_file}} python3 tools/crop-first/batch_subject_zones.py {{tasks}} {{root}}/vision-cache {{root}}/testing {{root}}/authorization.json {{output}} --workers {{workers}} --protocol label-quality

crop-record-label-progress root attention progress:
    python3 tools/crop-first/record_label_progress.py {{root}} {{attention}} {{progress}}

composition-freeze-real-attention root:
    python3 tools/composition/freeze_real_attention.py {{root}}

crop-analyze-corridor-reviews root output:
    python3 tools/crop-first/analyze_corridor_reviews.py {{root}} {{output}}

crop-prepare-corridor-review-batch python root native resizer encoder output:
    {{python}} tools/crop-first/prepare_corridor_review_batch.py {{root}} {{native}} {{resizer}} {{encoder}} {{output}}

crop-corridor-review-gallery python root analysis output:
    {{python}} tools/crop-first/corridor_review_gallery.py {{root}} {{analysis}} {{output}}

crop-corridor-browser root chromium scratch:
    python3 tools/crop-first/browser_check.py {{root}} {{chromium}} {{scratch}} --suite corridor

crop-freeze-label-pilot root output:
    python3 tools/crop-first/freeze_label_pilot.py {{root}} {{output}}

composition-audit-corridor-bank python cache pilot analysis output:
    {{python}} tools/composition/audit_corridor_bank.py {{cache}} {{pilot}} {{analysis}} {{output}}

composition-freeze-corridor-bank root:
    python3 tools/composition/freeze_corridor_bank.py {{root}}
