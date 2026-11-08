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

composition-owned-photo-proposals python root models split output:
    {{python}} tools/composition/owned_photo_proposals.py {{root}} {{models}} {{split}} {{output}}

composition-person-region-audit root output:
    python3 tools/composition/person_region_audit.py {{root}} {{output}}

composition-owned-screen-gallery root output asset_base:
    python3 tools/composition/owned_screen_gallery.py {{root}} {{output}} --asset-base {{asset_base}}

composition-svg-gutter-check python report output *args:
    {{python}} tools/composition/verify_svg_gutters.py {{report}} {{output}} {{args}}

composition-training-timing-plan cache output python:
    python3 tools/composition/training_timing_plan.py {{cache}} {{output}} --python {{python}}
