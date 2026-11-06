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
    cargo run {{CARGO_ARGS}} -p zensally-depth --example inspect_heic_depth -- "{{input}}" "{{output}}"

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
