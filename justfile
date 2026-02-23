# zensally development recipes

# Run tests (default features)
test:
    cargo test --workspace

# Run tests with all features
test-all:
    cargo test --workspace --features blazeface320

# Format code
fmt:
    cargo fmt --all

# Lint
clippy:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --features blazeface320 -- -D warnings

# Run benchmarks
bench *ARGS:
    cargo bench --package zensally-tract --bench face_bench {{ARGS}}

# Full CI check (local)
ci: fmt
    just clippy
    just test
    just test-all
    just bench -- --quick

# Download WIDER FACE validation dataset
download-wider-face:
    bash scripts/download_wider_face.sh

# Run WIDER FACE validation
validate:
    cargo run --package zensally-tract --example wider_validate --release

# Measure binary size of compressed models
model-sizes:
    @echo "ONNX model sizes:"
    @ls -la crates/zensally-tract/models/
