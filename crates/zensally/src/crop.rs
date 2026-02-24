//! Deprecated — smart crop logic has moved to `zenlayout::smart_crop`.
//!
//! This module is kept for backward compatibility. New code should use
//! `zenlayout::smart_crop::{SmartCropInput, FocusRect, HeatMap, compute_crop}` directly.
//!
//! To convert from detection types:
//! ```ignore
//! use zenlayout::smart_crop::{FocusRect, HeatMap, SmartCropInput};
//!
//! let input = SmartCropInput {
//!     focus_regions: faces.into_iter().map(|f| FocusRect {
//!         x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, weight: f.confidence,
//!     }).collect(),
//!     heatmap: saliency.map(|s| HeatMap {
//!         data: s.data, width: s.width, height: s.height,
//!     }),
//! };
//! let crops = input.compute_crops(1920, 1080, &targets);
//! ```
