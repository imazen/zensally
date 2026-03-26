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
//! Set `ZENTRACT_PLUGIN_PATH` to the full path of `libzentract_abi.so`.
//! Falls back to searching the directory containing the current executable.

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

/// Discover the zentract plugin path.
///
/// 1. `ZENTRACT_PLUGIN_PATH` env var (exact path)
/// 2. Same directory as the current executable
/// 3. System library search path (just the bare name)
pub fn discover_plugin() -> PathBuf {
    if let Ok(path) = std::env::var("ZENTRACT_PLUGIN_PATH") {
        return PathBuf::from(path);
    }

    // Try next to the executable
    if let Some(candidate) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join(plugin_filename())))
        .filter(|p| p.exists())
    {
        return candidate;
    }

    // Fall back to bare name (system library search)
    PathBuf::from(plugin_filename())
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
