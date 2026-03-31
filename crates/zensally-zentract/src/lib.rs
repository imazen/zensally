#![forbid(unsafe_code)]

//! Face detection and saliency via zentract ONNX plugin.
//!
//! Uses [`zentract_api`] to load ONNX models through a runtime plugin
//! (`libzentract_abi.so`), avoiding the 267-crate compile-time dependency
//! on tract-onnx. Preprocessing and postprocessing use the shared
//! implementations from [`zensally`] core.
//!
//! # Plugin Discovery
//!
//! The zentract plugin is found by searching, in order:
//!
//! 1. `ZENTRACT_PLUGIN_PATH` env var (exact path to .so/.dylib/.dll)
//! 2. Next to the current executable
//! 3. `target/release/` in the current working directory (dev builds)
//! 4. `../zentract/target/release/` (workspace sibling layout)
//! 5. System library search path (bare filename)
//!
//! To build the plugin:
//! ```sh
//! cd zentract && cargo build --release -p zentract-abi
//! ```
//!
//! Or use the justfile recipe in the zenfaces workspace:
//! ```sh
//! just build-plugin
//! ```

extern crate alloc;

#[cfg(feature = "ultraface")]
pub mod ultraface;

#[cfg(feature = "microsalnet")]
pub mod microsalnet;

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
pub mod analyzer;

#[cfg(feature = "ultraface")]
pub use ultraface::UltraFaceDetector;

#[cfg(feature = "microsalnet")]
pub use microsalnet::MicroSalNet;

#[cfg(all(feature = "ultraface", feature = "microsalnet"))]
pub use analyzer::ContentAnalyzer;

use std::path::PathBuf;

/// Discover the zentract plugin, searching multiple well-known locations.
///
/// Returns the first path that exists. If nothing is found, returns the
/// bare filename so `libloading` will try the system library search path.
///
/// See [module docs](self) for the full search order.
pub fn discover_plugin() -> PathBuf {
    // 1. Explicit env var
    if let Ok(path) = std::env::var("ZENTRACT_PLUGIN_PATH") {
        return PathBuf::from(path);
    }

    let filename = plugin_filename();

    // 2. Next to the running executable
    if let Some(p) = exe_sibling(filename) {
        return p;
    }

    // 3. target/release/ in cwd (dev builds)
    let cwd_target = PathBuf::from("target/release").join(filename);
    if cwd_target.exists() {
        return cwd_target;
    }

    // 4. Workspace sibling: ../zentract/target/release/
    if let Some(p) = workspace_sibling(filename) {
        return p;
    }

    // 5. Bare filename → system LD_LIBRARY_PATH / rpath
    PathBuf::from(filename)
}

/// Try to load the plugin, returning a clear error if not found.
///
/// Wraps [`discover_plugin`] + [`zentract_api::InferenceEngine::load`]
/// with a human-readable error message explaining how to build/install the plugin.
pub fn load_plugin() -> Result<zentract_api::InferenceEngine, anyhow::Error> {
    let path = discover_plugin();
    zentract_api::InferenceEngine::load(&path).map_err(|e| {
        if path
            .to_str()
            .map_or(false, |s| !s.contains('/') && !s.contains('\\'))
        {
            // Bare filename — nothing on disk matched
            anyhow::anyhow!(
                "zentract plugin not found.\n\
                 \n\
                 Build it with:\n\
                 \n\
                 \x20   cd zentract && cargo build --release -p zentract-abi\n\
                 \n\
                 Then either:\n\
                 \x20 • Set ZENTRACT_PLUGIN_PATH=/path/to/{filename}\n\
                 \x20 • Copy {filename} next to your binary\n\
                 \x20 • Symlink it into target/release/\n\
                 \n\
                 Underlying error: {e}",
                filename = plugin_filename(),
            )
        } else {
            anyhow::anyhow!(
                "failed to load zentract plugin from {}: {e}",
                path.display()
            )
        }
    })
}

fn plugin_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "zentract_abi.dll"
    } else if cfg!(target_os = "macos") {
        "libzentract_abi.dylib"
    } else {
        "libzentract_abi.so"
    }
}

fn exe_sibling(filename: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join(filename);
    candidate.exists().then_some(candidate)
}

fn workspace_sibling(filename: &str) -> Option<PathBuf> {
    // Walk up from cwd looking for a zentract/ sibling with a built plugin
    let cwd = std::env::current_dir().ok()?;
    for ancestor in cwd.ancestors().take(4) {
        let candidate = ancestor.join("zentract/target/release").join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Decompress a gzip-compressed model embedded via `include_bytes!`.
///
/// Reads the original size from the gzip ISIZE trailer (last 4 bytes).
pub(crate) fn decompress_gz(compressed: &[u8]) -> alloc::vec::Vec<u8> {
    let len = compressed.len();
    let orig_size = u32::from_le_bytes([
        compressed[len - 4],
        compressed[len - 3],
        compressed[len - 2],
        compressed[len - 1],
    ]) as usize;

    let mut decompressor = zenflate::Decompressor::new();
    let mut output = alloc::vec![0u8; orig_size];
    let outcome = decompressor
        .gzip_decompress(compressed, &mut output, enough::Unstoppable)
        .expect("embedded model decompression failed");
    debug_assert_eq!(outcome.output_written, orig_size);
    output
}
