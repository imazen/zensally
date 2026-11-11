//! Detached bench-harness internals.
//!
//! `#[path]`-includes the real crate-private modules under test so the
//! benchmarks exercise production `stream_prep`/`sat_crop` code without
//! widening `zensally`'s public API. The re-exports below keep the
//! modules' `crate::` paths resolving exactly as they do inside
//! `zensally`.

pub use zensally::{CropRect, FaceRect, FocusRegion, ImageRef, PixelFormat};

pub mod preprocess {
    pub use zensally::preprocess::*;
}

#[path = "../../../crates/zensally/src/stream_prep.rs"]
pub mod stream_prep;

#[path = "../../../crates/zensally/src/sat_crop.rs"]
pub mod sat_crop;
