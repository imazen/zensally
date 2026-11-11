# zensally development recipes

# Run tests (default features)
test:
    cargo test --workspace

# Run tests with all feature flags
test-all:
    cargo test --workspace --all-features

# Format code
fmt:
    cargo fmt -p zensally -p zensally-tract -p zensally-zentract

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
    cargo fmt -p zensally -p zensally-tract -p zensally-zentract -- --check
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
    cross test --workspace --all-features --target i686-unknown-linux-gnu

# Standalone native ORT experiment (not the Rust detector default).
native-ort-prepare output:
    python3 tools/ort-native/prepare.py "{{output}}"

native-ort-build +args:
    python3 tools/ort-native/build.py {{args}}

native-ort-windows +args:
    python3 tools/ort-native/windows.py {{args}}

native-ort-bench +args:
    python3 tools/ort-native/bench.py {{args}}

# Explicit dispatch only: builds and tests native Windows x64 and ARM64.
native-ort-ci-run:
    gh workflow run native-ort.yml --ref main

native-ort-ci-status run:
    gh run view {{run}} --json jobs --jq '.jobs[]|{name,id:.databaseId,status,conclusion,active:[.steps[]|select(.status=="in_progress")|.name]}'

native-ort-ci-logs run:
    gh run view {{run}} --log

# Native local GPU tests; choose an isolated Python environment via GPU_PYTHON.
native-ort-gpu +args:
    {{env_var("GPU_PYTHON")}} tools/ort-native/gpu.py {{args}}

native-ort-gpu-plan output fixtures:
    {{env_var("GPU_PYTHON")}} tools/ort-native/gpu-plan.py "{{output}}" "{{fixtures}}"

native-ort-gpu-pair plan output:
    {{env_var_or_default("ZENBENCH_BIN", "zenbench")}} pair "{{plan}}" --output "{{output}}"

# Profiles are separate from paired timing. Output directory must be fresh.
native-ort-gpu-profile provider output model input:
    mkdir -p "{{output}}"
    {{env_var_or_default("NSYS_BIN", "nsys")}} profile --trace=cuda,vulkan --sample=none --cpuctxsw=none --cuda-memory-usage=true --stats=false --output="{{output}}/trace" {{env_var("GPU_PYTHON")}} tools/ort-native/gpu.py {{provider}} "{{output}}/outputs" --model "{{model}}" --input "{{input}}" --iterations 100
