# zensally development recipes

# Run tests (default features)
test:
    cargo test --workspace

# Run tests with all feature flags
test-all:
    cargo test --workspace --features "blazeface320,mediapipe,yunet,analyzer"

# Format code
fmt:
    cargo fmt --all

# Lint
clippy:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --features "blazeface320,mediapipe,yunet,analyzer" -- -D warnings

# Check all feature permutations
feature-check:
    cargo check --workspace
    cargo check --workspace --features blazeface320
    cargo check --workspace --features mediapipe
    cargo check --workspace --features yunet
    cargo check --workspace --features analyzer
    cargo check --workspace --features "ultraface,microsalnet"
    cargo check --workspace --features "blazeface320,mediapipe,yunet,analyzer"

# Run benchmarks
bench *ARGS:
    cargo bench --package zensally-tract --bench face_bench {{ARGS}}

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
