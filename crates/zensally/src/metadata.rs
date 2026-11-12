//! Model-agnostic, resolution-independent smart-crop metadata (`CropMetadata`)
//! with zero-dependency RFC 8949 CBOR and RFC 8259 JSON serialization.
//!
//! `CropMetadata` decouples neural/classical visual analysis from crop search:
//! run any analyzer once (`CropAnalyzerV2`, `MicroCropAnalyzer`, or a future
//! detector/segmenter), serialize `CropMetadata` into an image container box
//! (JXL, PNG, WebP, AVIF, XMP) or database column, and later derive any number
//! of aspect ratios, zoom modes (`Minimal` / `Maximal`), target resolutions,
//! or user focus overrides in ~30–50 µs with zero image decoding and zero
//! neural inference.
//!
//! # Third-Party Interoperability (CBOR & JSON)
//!
//! Both encodings use standard, self-describing map/object keys rather than
//! positional binary headers, so third-party tools in any language can read
//! or author them directly with off-the-shelf libraries:
//!
//! - **RFC 8949 CBOR** ([`CropMetadata::to_cbor`] / [`CropMetadata::from_cbor`]):
//!   encodes the root as a standard CBOR map (`Major Type 5`) with UTF-8 string
//!   keys (`"v"`, `"producer"`, `"aspect"`, `"subjects"`, `"saliency"`,
//!   `"foreground"`, `"structural_energy"`, `"horizon_profile"`). Each 2D
//!   [`CuePlane`] (`{"w": width, "h": height, "q8": <bytes>}`) stores its pixels
//!   as a native CBOR byte string (`Major Type 2`, 1 byte per cell) — decoding
//!   directly to `bytes` in Python (`cbor2.loads`), `Uint8Array` in JavaScript
//!   (`cborg` / `cbor-x`), or `[]byte` in Go (`fxamacker/cbor`).
//!   - Full `64×48` 3-plane metadata: **~9.3 KB** raw CBOR (**~1.8 KB** Brotli/Zstd).
//!   - Compact `32×24` metadata ([`CropMetadata::into_compact_32x24`]):
//!     **~2.4 KB** raw CBOR (**~650 B** Brotli/Zstd).
//! - **JSON** ([`CropMetadata::to_json`] / [`CropMetadata::from_json`], plus
//!   `serde::Serialize` / `serde::Deserialize` under `feature = "serde"`):
//!   uses the same structure with standard Base64 (`"q8_b64"`) for [`CuePlane`]
//!   data (~3.3 KB at `32×24`, ~12.5 KB at `64×48`). The decoder accepts both
//!   `"q8_b64"` (Base64 string) and `"q8"` (integer array `[0..=255]`).

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::sat_crop::{
    CropInputSources, DualIntegralMap, FaceCellBox, FaceLandmarkInput, MATTE_GRADIENT_WEIGHT,
    SAT_SCALE, SatCropConfig, SatCropMode, TargetToSrc, add_rect, bbox_above, ceil32, raise_rect,
    resample_area_avg, solve_crop_sat,
};
use crate::stream_prep::{floor32, round32};
use crate::{CropRect, FocusRegion};

/// Schema version emitted by [`CropMetadata`].
pub const CROP_METADATA_VERSION: u32 = 1;

/// RFC 8949 §3.4.6 self-described CBOR tag (`55799`), accepted transparently by
/// [`CropMetadata::from_cbor`].
pub const CBOR_SELF_DESCRIBE_TAG: u64 = 55799;

/// Default solver grid width when reconstructing a [`DualIntegralMap`].
pub const DEFAULT_SOLVER_GRID_W: usize = 64;

/// Default solver grid height when reconstructing a [`DualIntegralMap`].
pub const DEFAULT_SOLVER_GRID_H: usize = 48;

/// Default `subject_threshold` fraction of peak semantic energy.
pub const DEFAULT_SUBJECT_THRESHOLD: f32 = 0.25;

/// `f32::sqrt` for `no_std` (IEEE-754 initial guess + 4 Newton-Raphson steps).
#[inline]
fn sqrt32(x: f32) -> f32 {
    if x <= 0.0 || x.is_nan() {
        return 0.0;
    }
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    0.5 * (y + x / y)
}

#[inline]
fn quantize_u8(v: f32) -> u8 {
    round32(v.clamp(0.0, 1.0) * 255.0) as u8
}

#[inline]
fn dequantize_u8(q: u8) -> f32 {
    q as f32 * (1.0 / 255.0)
}

// ---------------------------------------------------------------------------
// Data Model
// ---------------------------------------------------------------------------

/// Semantic category of a detected [`SubjectRegion`].
///
/// Serializes as a lowercase string (`"face"`, `"person"`, `"pet"`, `"product"`,
/// `"text"`, `"focal_point"`, or any custom tag) so future detectors can add new
/// classes without breaking existing parsers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubjectKind {
    /// Human face (`"face"`).
    Face,
    /// Human figure / upper body / person (`"person"`).
    Person,
    /// Animal / pet (`"pet"`).
    Pet,
    /// Product / foreground object (`"product"`).
    Product,
    /// Text / signage / logo (`"text"`).
    Text,
    /// Editorial or user-specified focal point (`"focal_point"`).
    FocalPoint,
    /// Forward-compatible custom subject class.
    Custom(String),
}

impl SubjectKind {
    /// Canonical lowercase string tag for CBOR and JSON serialization.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Face => "face",
            Self::Person => "person",
            Self::Pet => "pet",
            Self::Product => "product",
            Self::Text => "text",
            Self::FocalPoint => "focal_point",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parse a string tag into a [`SubjectKind`].
    pub fn from_str_tag(tag: &str) -> Self {
        match tag {
            "face" => Self::Face,
            "person" => Self::Person,
            "pet" => Self::Pet,
            "product" => Self::Product,
            "text" => Self::Text,
            "focal_point" | "focus" => Self::FocalPoint,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// A semantic subject region in resolution-independent normalized `[0.0, 1.0]`
/// image coordinates (`(0, 0)` = top-left, `(1, 1)` = bottom-right).
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectRegion {
    /// Semantic category (`"face"`, `"person"`, `"pet"`, etc.).
    pub kind: SubjectKind,
    /// Detector confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Outer bounding box `[x1, y1, x2, y2]` in normalized `[0.0, 1.0]` image coords.
    pub bbox: [f32; 4],
    /// Inner anatomical or semantic core `[x1, y1, x2, y2]` in `[0.0, 1.0]` that
    /// crop borders must strongly avoid slicing (e.g., eyebrow-to-mouth core).
    pub protected_core: Option<[f32; 4]>,
    /// Tightest inviolable feature band `[x1, y1, x2, y2]` in `[0.0, 1.0]`
    /// (e.g., raw eye-to-mouth landmark span) used when `protected_core` cannot
    /// fit along an extreme aspect axis.
    pub feature_band: Option<[f32; 4]>,
    /// Preferred framing margin `[x1, y1, x2, y2]` in normalized image coords
    /// (may extend slightly outside `0.0..=1.0`, e.g. forehead headroom above a
    /// top-edge face or chin/neck clearance below).
    pub preferred_margin: Option<[f32; 4]>,
    /// Optional `[x, y]` keypoints in normalized `[0.0, 1.0]` image coords
    /// (e.g. 5-point facial landmarks `[right_eye, left_eye, nose, right_mouth, left_mouth]`).
    pub keypoints: Vec<[f32; 2]>,
}

impl SubjectRegion {
    /// Construct a basic [`SubjectRegion`] from a kind, normalized `[x1, y1, x2, y2]`
    /// bounding box, and confidence.
    pub fn new(kind: SubjectKind, bbox: [f32; 4], confidence: f32) -> Self {
        Self {
            kind,
            confidence: confidence.clamp(0.0, 1.0),
            bbox,
            protected_core: None,
            feature_band: None,
            preferred_margin: None,
            keypoints: Vec::new(),
        }
    }

    /// Convert a [`FaceLandmarkInput`] (percentage `0.0..100.0` coordinates) into a
    /// resolution-independent [`SubjectRegion`] with precomputed anatomical
    /// `protected_core`, `feature_band`, and `preferred_margin`.
    pub fn from_face_landmark(f: &FaceLandmarkInput) -> Option<Self> {
        let fx1 = (f.rect.x1 / 100.0).clamp(0.0, 1.0);
        let fy1 = (f.rect.y1 / 100.0).clamp(0.0, 1.0);
        let fx2 = (f.rect.x2 / 100.0).clamp(0.0, 1.0);
        let fy2 = (f.rect.y2 / 100.0).clamp(0.0, 1.0);
        let (fw, fh) = (fx2 - fx1, fy2 - fy1);
        if fw <= 0.0 || fh <= 0.0 {
            return None;
        }
        let conf = f.rect.confidence.clamp(0.0, 1.0);
        let headroom_y1 = fy1 - 0.35 * fh;
        let chin_y2 = fy2 + 0.25 * fh;

        let (px1, py1, px2, py2, bx1, by1, bx2, by2, keypoints);
        if let Some(lm) = f.landmarks {
            let mut lx1 = f32::MAX;
            let mut ly1 = f32::MAX;
            let mut lx2 = f32::MIN;
            let mut ly2 = f32::MIN;
            let mut kps = Vec::with_capacity(5);
            for p in &lm {
                let kx = (p[0] / 100.0).clamp(0.0, 1.0);
                let ky = (p[1] / 100.0).clamp(0.0, 1.0);
                lx1 = lx1.min(kx);
                ly1 = ly1.min(ky);
                lx2 = lx2.max(kx);
                ly2 = ly2.max(ky);
                kps.push([kx, ky]);
            }
            let eye_y = kps[0][1].min(kps[1][1]);
            let mouth_y = kps[3][1].max(kps[4][1]);
            let eye_span = (lx2 - lx1).max(0.30 * fw);
            px1 = (lx1 - 0.30 * eye_span)
                .max(fx1 + 0.06 * fw)
                .min(fx2 - 0.06 * fw);
            px2 = (lx2 + 0.30 * eye_span).min(fx2 - 0.06 * fw);
            py1 = (eye_y - 0.18 * fh).max(fy1 + 0.04 * fh).clamp(0.0, 1.0);
            py2 = (mouth_y + 0.08 * fh).min(fy2).clamp(0.0, 1.0);
            bx1 = lx1.clamp(0.0, 1.0);
            by1 = ly1.clamp(0.0, 1.0);
            bx2 = lx2.clamp(0.0, 1.0);
            by2 = ly2.clamp(0.0, 1.0);
            keypoints = kps;
        } else {
            px1 = fx1 + 0.12 * fw;
            px2 = fx2 - 0.12 * fw;
            py1 = (fy1 + 0.15 * fh).clamp(0.0, 1.0);
            py2 = (fy2 + 0.15 * fh).clamp(0.0, 1.0);
            bx1 = px1;
            by1 = py1;
            bx2 = px2;
            by2 = py2;
            keypoints = Vec::new();
        }

        Some(Self {
            kind: SubjectKind::Face,
            confidence: conf,
            bbox: [fx1, fy1, fx2, fy2],
            protected_core: Some([px1, py1, px2, py2]),
            feature_band: Some([bx1, by1, bx2, by2]),
            preferred_margin: Some([fx1, headroom_y1, fx2, chin_y2]),
            keypoints,
        })
    }

    /// Convert a [`FocusRegion`] (percentage `0.0..100.0` coordinates) into a
    /// [`SubjectKind::FocalPoint`] [`SubjectRegion`].
    pub fn from_focus_region(r: &FocusRegion) -> Option<Self> {
        let x1 = (r.x1 / 100.0).clamp(0.0, 1.0);
        let y1 = (r.y1 / 100.0).clamp(0.0, 1.0);
        let x2 = (r.x2 / 100.0).clamp(0.0, 1.0);
        let y2 = (r.y2 / 100.0).clamp(0.0, 1.0);
        if x2 <= x1 || y2 <= y1 {
            return None;
        }
        Some(Self {
            kind: SubjectKind::FocalPoint,
            confidence: 1.0,
            bbox: [x1, y1, x2, y2],
            protected_core: Some([x1, y1, x2, y2]),
            feature_band: Some([x1, y1, x2, y2]),
            preferred_margin: Some([x1, y1, x2, y2]),
            keypoints: Vec::new(),
        })
    }
}

/// A 2D spatial cue plane over the normalized `[0.0, 1.0] × [0.0, 1.0]` image
/// domain, quantized to 8 bits per cell (`0..=255` mapping linearly to `0.0..=1.0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuePlane {
    /// Grid width in cells.
    pub width: u16,
    /// Grid height in cells.
    pub height: u16,
    /// Row-major quantized values (`width * height` bytes).
    pub data: Vec<u8>,
}

impl CuePlane {
    /// Create a [`CuePlane`] by quantizing row-major `f32` values in `[0.0, 1.0]`
    /// to `0..=255`.
    pub fn from_f32_slice(width: u16, height: u16, values: &[f32]) -> Self {
        let n = width as usize * height as usize;
        assert_eq!(values.len(), n, "CuePlane values len != {width}x{height}");
        let mut data = Vec::with_capacity(n);
        for &v in values {
            data.push(quantize_u8(v));
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Validate that `data.len() == width * height`.
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.data.len() == self.width as usize * self.height as usize
    }

    /// Dequantize to a `Vec<f32>` in `[0.0, 1.0]` at the plane's native dimensions.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().copied().map(dequantize_u8).collect()
    }

    /// Decode this plane onto a `target_w × target_h` solver grid in `[0.0, 1.0]`.
    ///
    /// Uses a direct `u8 -> f32` table/multiply when dimensions match, or
    /// bilinear interpolation when resampling (e.g. expanding a compact `32×24`
    /// plane onto the `64×48` solver grid).
    pub fn decode_into_grid(&self, target_w: usize, target_h: usize, out: &mut [f32]) {
        assert_eq!(out.len(), target_w * target_h);
        let sw = self.width as usize;
        let sh = self.height as usize;
        if sw == 0 || sh == 0 || self.data.len() < sw * sh || target_w == 0 || target_h == 0 {
            out.fill(0.0);
            return;
        }
        if sw == target_w && sh == target_h {
            for (dst, &q) in out.iter_mut().zip(self.data.iter()) {
                *dst = dequantize_u8(q);
            }
            return;
        }
        for ty in 0..target_h {
            let fy = ((ty as f32 + 0.5) * (sh as f32 / target_h as f32) - 0.5)
                .clamp(0.0, (sh - 1) as f32);
            let y0 = floor32(fy) as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let wy = fy - y0 as f32;
            for tx in 0..target_w {
                let fx = ((tx as f32 + 0.5) * (sw as f32 / target_w as f32) - 0.5)
                    .clamp(0.0, (sw - 1) as f32);
                let x0 = floor32(fx) as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let wx = fx - x0 as f32;

                let v00 = dequantize_u8(self.data[y0 * sw + x0]);
                let v01 = dequantize_u8(self.data[y0 * sw + x1]);
                let v10 = dequantize_u8(self.data[y1 * sw + x0]);
                let v11 = dequantize_u8(self.data[y1 * sw + x1]);

                let top = v00 + wx * (v01 - v00);
                let bot = v10 + wx * (v11 - v10);
                out[ty * target_w + tx] = top + wy * (bot - top);
            }
        }
    }

    /// Downsample this plane to at most `dst_w × dst_h` using area averaging.
    pub fn downsample(&self, dst_w: u16, dst_h: u16) -> Self {
        if self.width <= dst_w && self.height <= dst_h {
            return self.clone();
        }
        let sw = self.width as usize;
        let sh = self.height as usize;
        let dw = dst_w.max(1) as usize;
        let dh = dst_h.max(1) as usize;
        if sw == 0 || sh == 0 || self.data.len() < sw * sh {
            return Self {
                width: dst_w,
                height: dst_h,
                data: vec![0; dw * dh],
            };
        }
        // Fast exact 2×2 box filter when halving both axes.
        if sw == dw * 2 && sh == dh * 2 {
            let mut data = Vec::with_capacity(dw * dh);
            for y in 0..dh {
                let r0 = (2 * y) * sw;
                let r1 = (2 * y + 1) * sw;
                for x in 0..dw {
                    let c = 2 * x;
                    let sum = self.data[r0 + c] as u16
                        + self.data[r0 + c + 1] as u16
                        + self.data[r1 + c] as u16
                        + self.data[r1 + c + 1] as u16;
                    data.push(((sum + 2) >> 2) as u8);
                }
            }
            return Self {
                width: dst_w,
                height: dst_h,
                data,
            };
        }

        let src_f32 = self.to_f32_vec();
        let to_src = TargetToSrc::stretch(sw as u32, sh as u32, dw as u32 * 8, dh as u32 * 8);
        let mut out_f32 = vec![0.0f32; dw * dh];
        resample_area_avg(
            &src_f32,
            sw as u32,
            sh as u32,
            &to_src,
            dw,
            dh,
            &mut out_f32,
        );
        Self::from_f32_slice(dst_w, dst_h, &out_f32)
    }
}

/// Model-agnostic, resolution-independent smart-crop metadata.
///
/// Can be embedded in image containers (JXL/PNG/WebP/AVIF boxes, XMP) or stored
/// in a database as RFC 8949 CBOR ([`Self::to_cbor`]) or JSON ([`Self::to_json`])
/// for ~30–50 µs future crop derivation at any aspect ratio or resized dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct CropMetadata {
    /// Schema version (currently [`CROP_METADATA_VERSION`] = `1`).
    pub version: u32,
    /// Optional producer identifier (e.g. `"zensally/0.2.0"` or `"zensally-micro/0.2.0"`).
    pub producer: Option<String>,
    /// Optional source image aspect ratio (`width / height`).
    pub aspect_ratio: Option<f32>,
    /// Semantic subject regions in normalized `[0.0, 1.0]` image coordinates.
    pub subjects: Vec<SubjectRegion>,
    /// Normalized peak-squared visual saliency plane in `[0.0, 1.0]`.
    pub saliency: Option<CuePlane>,
    /// Unified foreground silhouette / matte probability plane in `[0.0, 1.0]`.
    pub foreground: Option<CuePlane>,
    /// Percentile-normalized structural edge/chroma energy plane in `[0.0, 1.0]`.
    pub structural_energy: Option<CuePlane>,
    /// Per-row horizontal continuity profile quantized to `0..=255` (`[0.0, 1.0]`).
    pub horizon_profile: Option<Vec<u8>>,
    /// Fraction of peak semantic energy used to form the unanchored subject bbox
    /// (default [`DEFAULT_SUBJECT_THRESHOLD`] = `0.25`).
    pub subject_threshold: f32,
}

impl Default for CropMetadata {
    fn default() -> Self {
        Self {
            version: CROP_METADATA_VERSION,
            producer: None,
            aspect_ratio: None,
            subjects: Vec::new(),
            saliency: None,
            foreground: None,
            structural_energy: None,
            horizon_profile: None,
            subject_threshold: DEFAULT_SUBJECT_THRESHOLD,
        }
    }
}

impl CropMetadata {
    /// Extract normalized, model-agnostic [`CropMetadata`] from raw analyzer
    /// [`CropInputSources`].
    ///
    /// Performs the same peak-normalized saliency squaring, saliency-supported
    /// foreground matte fusion, p95 classical energy normalization, and facial
    /// landmark core derivation as [`DualIntegralMap::from_crop_inputs`].
    pub fn from_crop_inputs(
        sources: &CropInputSources<'_>,
        gw: usize,
        gh: usize,
        to_src: &TargetToSrc,
        subject_threshold: f32,
        producer: Option<&str>,
    ) -> Self {
        let n = gw * gh;
        if let Some(e) = sources.classical_energy {
            assert!(e.len() >= n, "classical_energy len < grid {gw}x{gh}");
        }
        if let Some(h) = sources.horizon_profile {
            assert_eq!(h.len(), gh, "horizon_profile len != {gh}");
        }
        let (src_w, src_h) = (to_src.src_w as f64, to_src.src_h as f64);

        // Stretch mapping over `[0, src_w] × [0, src_h]` so stored planes always
        // cover the normalized `[0.0, 1.0] × [0.0, 1.0]` source image domain.
        let norm_to_src =
            TargetToSrc::stretch(to_src.src_w, to_src.src_h, gw as u32 * 8, gh as u32 * 8);

        let cell_active = |cx: usize, cy: usize| -> bool {
            let sx1 = to_src.map_x(cx as f64 * 8.0);
            let sx2 = to_src.map_x((cx + 1) as f64 * 8.0);
            let sy1 = to_src.map_y(cy as f64 * 8.0);
            let sy2 = to_src.map_y((cy + 1) as f64 * 8.0);
            sx2 > 0.0 && sy2 > 0.0 && sx1 < src_w && sy1 < src_h
        };

        // 1. Saliency: resample to normalized image grid, peak-normalize, and square.
        let mut sal_sq = vec![0.0f32; n];
        let has_sal = sources.saliency.is_some();
        if let Some(sal) = sources.saliency {
            resample_area_avg(
                &sal.data,
                sal.width,
                sal.height,
                &norm_to_src,
                gw,
                gh,
                &mut sal_sq,
            );
            let mut vals: Vec<f32> = sal_sq.iter().copied().filter(|&v| v > 0.0).collect();
            let peak = if vals.is_empty() {
                0.0
            } else {
                vals.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let k = vals.len().div_ceil(200).max(8).min(vals.len());
                vals[vals.len() - k..].iter().sum::<f32>() / k as f32
            };
            if peak > 0.0 {
                let norm = peak.max(0.15);
                for v in &mut sal_sq {
                    let s = (*v / norm).clamp(0.0, 1.0);
                    *v = s * s;
                }
            } else {
                for v in &mut sal_sq {
                    let s = v.clamp(0.0, 1.0);
                    *v = s * s;
                }
            }
        }

        // 2. Matte: resample to normalized image grid.
        let mut matte = vec![0.0f32; n];
        let has_matte = sources.matte.is_some();
        if let Some(m) = sources.matte {
            resample_area_avg(&m.data, m.width, m.height, &norm_to_src, gw, gh, &mut matte);
            for v in &mut matte {
                *v = v.clamp(0.0, 1.0);
            }
        }

        // 3. Classical structural energy: p95-normalized to `[0.0, 1.0]` so the
        // quantized u8 plane uses the full 0..=255 range (scaled by 0.15 on SAT build).
        let structural_energy = if let Some(e) = sources.classical_energy {
            let mut norm_e = vec![0.0f32; n];
            let mut vals: Vec<f32> = (0..n)
                .filter(|&i| cell_active(i % gw, i / gw) && e[i] > 0.0)
                .map(|i| e[i])
                .collect();
            if !vals.is_empty() {
                vals.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let p95 = vals[(vals.len() * 95 / 100).min(vals.len() - 1)];
                if p95 > 0.0 {
                    for i in 0..n {
                        if cell_active(i % gw, i / gw) {
                            norm_e[i] = (e[i].max(0.0) / p95).min(1.0);
                        }
                    }
                }
            }
            Some(CuePlane::from_f32_slice(gw as u16, gh as u16, &norm_e))
        } else {
            None
        };

        // 4. Subjects (faces + manual focus).
        let mut subjects = Vec::with_capacity(sources.faces.len() + sources.manual_focus.len());
        for f in sources.faces {
            if let Some(region) = SubjectRegion::from_face_landmark(f) {
                subjects.push(region);
            }
        }
        for r in sources.manual_focus {
            if let Some(region) = SubjectRegion::from_focus_region(r) {
                subjects.push(region);
            }
        }

        // 5. Unified foreground plane `fg`.
        let mut fg = vec![0.0f32; n];
        for (g, &s) in fg.iter_mut().zip(sal_sq.iter()) {
            *g = ((s - 0.12) / 0.60).clamp(0.0, 1.0);
        }
        if has_matte {
            if has_sal {
                let mut support = vec![0.0f32; n];
                for (v, &s) in support.iter_mut().zip(sal_sq.iter()) {
                    *v = sqrt32(s);
                }
                for subj in &subjects {
                    let fx1 = subj.bbox[0] * gw as f32;
                    let fy1 = subj.bbox[1] * gh as f32;
                    let fx2 = subj.bbox[2] * gw as f32;
                    let fy2 = subj.bbox[3] * gh as f32;
                    let (fw, fh) = (fx2 - fx1, fy2 - fy1);
                    if subj.kind == SubjectKind::FocalPoint {
                        raise_rect(&mut support, gw, gh, fx1, fy1, fx2, fy2, 1.0);
                    } else {
                        let headroom_y1 = subj
                            .preferred_margin
                            .map_or(fy1 - 0.35 * fh, |m| m[1] * gh as f32);
                        raise_rect(
                            &mut support,
                            gw,
                            gh,
                            fx1 - 0.35 * fw,
                            headroom_y1,
                            fx2 + 0.35 * fw,
                            fy2 + 1.25 * fh,
                            1.0,
                        );
                    }
                }
                for i in 0..n {
                    let m = matte[i] * (0.20 + 0.80 * support[i]);
                    if m > fg[i] {
                        fg[i] = m;
                    }
                }
            } else {
                for i in 0..n {
                    if matte[i] > fg[i] {
                        fg[i] = matte[i];
                    }
                }
            }
        }

        let saliency_plane = if has_sal {
            Some(CuePlane::from_f32_slice(gw as u16, gh as u16, &sal_sq))
        } else {
            None
        };
        let foreground_plane = if has_sal || has_matte {
            Some(CuePlane::from_f32_slice(gw as u16, gh as u16, &fg))
        } else {
            None
        };
        let horizon_profile = sources
            .horizon_profile
            .map(|h| h.iter().copied().map(quantize_u8).collect());
        let aspect_ratio = if to_src.src_w > 0 && to_src.src_h > 0 {
            Some(to_src.src_w as f32 / to_src.src_h as f32)
        } else {
            None
        };

        Self {
            version: CROP_METADATA_VERSION,
            producer: producer.map(ToString::to_string),
            aspect_ratio,
            subjects,
            saliency: saliency_plane,
            foreground: foreground_plane,
            structural_energy,
            horizon_profile,
            subject_threshold,
        }
    }

    /// Downsample all 2D cue planes to at most `max_w × max_h` and the horizon
    /// profile to at most `max_h`.
    pub fn into_compact(mut self, max_w: u16, max_h: u16) -> Self {
        if let Some(p) = self.saliency.take() {
            self.saliency = Some(p.downsample(max_w, max_h));
        }
        if let Some(p) = self.foreground.take() {
            self.foreground = Some(p.downsample(max_w, max_h));
        }
        if let Some(p) = self.structural_energy.take() {
            self.structural_energy = Some(p.downsample(max_w, max_h));
        }
        if let Some(h) = self.horizon_profile.take() {
            let target_h = max_h.max(1) as usize;
            if h.len() == target_h * 2 {
                let mut down = Vec::with_capacity(target_h);
                for y in 0..target_h {
                    let s = h[2 * y] as u16 + h[2 * y + 1] as u16;
                    down.push(((s + 1) >> 1) as u8);
                }
                self.horizon_profile = Some(down);
            } else if h.len() > target_h {
                let mut down = Vec::with_capacity(target_h);
                for y in 0..target_h {
                    let idx = (y * h.len()) / target_h;
                    down.push(h[idx]);
                }
                self.horizon_profile = Some(down);
            } else {
                self.horizon_profile = Some(h);
            }
        }
        self
    }

    /// Convenience helper: downsample 2D planes from `64×48` to `32×24`
    /// (**~2.4 KB** raw CBOR, **~650 B** compressed) while retaining full
    /// floating-point precision on all [`SubjectRegion`] coordinates.
    pub fn into_compact_32x24(self) -> Self {
        self.into_compact(32, 24)
    }

    /// Reconstruct a [`DualIntegralMap`] and [`TargetToSrc`] for a target source
    /// resolution `(src_w, src_h)` and optional additional runtime `manual_focus`
    /// regions, using the default `64×48` solver grid.
    ///
    /// Runs in ~15 µs with zero image decoding and zero neural inference.
    pub fn build_integral_map(
        &self,
        src_w: u32,
        src_h: u32,
        manual_focus: &[FocusRegion],
    ) -> (DualIntegralMap, TargetToSrc) {
        self.build_integral_map_with_grid(
            src_w,
            src_h,
            manual_focus,
            DEFAULT_SOLVER_GRID_W,
            DEFAULT_SOLVER_GRID_H,
        )
    }

    /// Reconstruct a [`DualIntegralMap`] and [`TargetToSrc`] on an explicit
    /// `gw × gh` solver grid.
    pub fn build_integral_map_with_grid(
        &self,
        src_w: u32,
        src_h: u32,
        manual_focus: &[FocusRegion],
        gw: usize,
        gh: usize,
    ) -> (DualIntegralMap, TargetToSrc) {
        let gw = gw.max(1);
        let gh = gh.max(1);
        let n = gw * gh;
        let to_src = TargetToSrc::stretch(src_w, src_h, gw as u32 * 8, gh as u32 * 8);

        let mut sal_sq = vec![0.0f32; n];
        if let Some(ref p) = self.saliency {
            p.decode_into_grid(gw, gh, &mut sal_sq);
        }

        let mut fg = vec![0.0f32; n];
        if let Some(ref p) = self.foreground {
            p.decode_into_grid(gw, gh, &mut fg);
        } else if self.saliency.is_some() {
            for (g, &s) in fg.iter_mut().zip(sal_sq.iter()) {
                *g = ((s - 0.12) / 0.60).clamp(0.0, 1.0);
            }
        }

        let mut classical = vec![0.0f32; n];
        if let Some(ref p) = self.structural_energy {
            p.decode_into_grid(gw, gh, &mut classical);
            for v in &mut classical {
                *v *= 0.15;
            }
        }

        let mut landmark_w = vec![0.0f32; n];
        let mut cut = vec![0.0f32; n];
        let mut faces: Vec<FaceCellBox> = Vec::new();
        let mut focus_boxes: Vec<FaceCellBox> = Vec::new();

        for subj in &self.subjects {
            let fx1 = (subj.bbox[0] * gw as f32).clamp(0.0, gw as f32);
            let fy1 = (subj.bbox[1] * gh as f32).clamp(0.0, gh as f32);
            let fx2 = (subj.bbox[2] * gw as f32).clamp(0.0, gw as f32);
            let fy2 = (subj.bbox[3] * gh as f32).clamp(0.0, gh as f32);
            let (fw, fh) = (fx2 - fx1, fy2 - fy1);
            if fw <= 0.0 || fh <= 0.0 {
                continue;
            }
            let conf = subj.confidence.clamp(0.0, 1.0);

            if subj.kind == SubjectKind::FocalPoint {
                add_rect(&mut landmark_w, gw, gh, fx1, fy1, fx2, fy2, 12.0 * conf);
                focus_boxes.push(FaceCellBox {
                    fx1,
                    fy1,
                    fx2,
                    fy2,
                    headroom_y1: fy1,
                    chin_y2: fy2,
                    px1: fx1,
                    py1: fy1,
                    px2: fx2,
                    py2: fy2,
                    bx1: fx1,
                    by1: fy1,
                    bx2: fx2,
                    by2: fy2,
                    confidence: conf,
                });
                continue;
            }

            let (headroom_y1, chin_y2) = if let Some(m) = subj.preferred_margin {
                (m[1] * gh as f32, m[3] * gh as f32)
            } else if subj.kind == SubjectKind::Face {
                (fy1 - 0.35 * fh, fy2 + 0.25 * fh)
            } else {
                (fy1 - 0.10 * fh, fy2 + 0.10 * fh)
            };

            let (px1, py1, px2, py2) = if let Some(c) = subj.protected_core {
                (
                    (c[0] * gw as f32).clamp(0.0, gw as f32),
                    (c[1] * gh as f32).clamp(0.0, gh as f32),
                    (c[2] * gw as f32).clamp(0.0, gw as f32),
                    (c[3] * gh as f32).clamp(0.0, gh as f32),
                )
            } else {
                (
                    fx1 + 0.12 * fw,
                    (fy1 + 0.15 * fh).clamp(0.0, gh as f32),
                    fx2 - 0.12 * fw,
                    (fy2 - 0.15 * fh).max(fy1 + 0.15 * fh).clamp(0.0, gh as f32),
                )
            };

            let (bx1, by1, bx2, by2) = if let Some(b) = subj.feature_band {
                (
                    (b[0] * gw as f32).clamp(0.0, gw as f32),
                    (b[1] * gh as f32).clamp(0.0, gh as f32),
                    (b[2] * gw as f32).clamp(0.0, gw as f32),
                    (b[3] * gh as f32).clamp(0.0, gh as f32),
                )
            } else {
                (px1, py1, px2, py2)
            };

            add_rect(&mut landmark_w, gw, gh, fx1, fy1, fx2, fy2, 6.0 * conf);
            if !subj.keypoints.is_empty() || subj.feature_band.is_some() {
                // Only add the +10*conf inner feature landmark boost when
                // keypoints are present or when the feature band differs from
                // the un-landmarked fallback core.
                let has_distinct_band =
                    !subj.keypoints.is_empty() || subj.feature_band != subj.protected_core;
                if has_distinct_band {
                    add_rect(&mut landmark_w, gw, gh, bx1, by1, bx2, by2, 10.0 * conf);
                }
            }
            if subj.kind == SubjectKind::Face {
                add_rect(
                    &mut landmark_w,
                    gw,
                    gh,
                    fx1 - 0.35 * fw,
                    fy2,
                    fx2 + 0.35 * fw,
                    fy2 + 1.25 * fh,
                    1.2 * conf,
                );
            }
            add_rect(&mut cut, gw, gh, fx1, headroom_y1, fx2, chin_y2, 8.0 * conf);
            add_rect(&mut cut, gw, gh, px1, py1, px2, py2, 45.0 * conf);

            faces.push(FaceCellBox {
                fx1,
                fy1,
                fx2,
                fy2,
                headroom_y1,
                chin_y2,
                px1,
                py1,
                px2,
                py2,
                bx1,
                by1,
                bx2,
                by2,
                confidence: conf,
            });
        }

        // Additional runtime manual focus regions passed at crop time.
        for r in manual_focus {
            let fx1 = (r.x1 / 100.0 * gw as f32).clamp(0.0, gw as f32);
            let fy1 = (r.y1 / 100.0 * gh as f32).clamp(0.0, gh as f32);
            let fx2 = (r.x2 / 100.0 * gw as f32).clamp(0.0, gw as f32);
            let fy2 = (r.y2 / 100.0 * gh as f32).clamp(0.0, gh as f32);
            if fx2 <= fx1 || fy2 <= fy1 {
                continue;
            }
            add_rect(&mut landmark_w, gw, gh, fx1, fy1, fx2, fy2, 12.0);
            focus_boxes.push(FaceCellBox {
                fx1,
                fy1,
                fx2,
                fy2,
                headroom_y1: fy1,
                chin_y2: fy2,
                px1: fx1,
                py1: fy1,
                px2: fx2,
                py2: fy2,
                bx1: fx1,
                by1: fy1,
                bx2: fx2,
                by2: fy2,
                confidence: 1.0,
            });
        }

        let cell_cap_fixed = u32::MAX / n as u32;
        let cell_cap = cell_cap_fixed as f32 / SAT_SCALE;
        let to_fixed = |v: f32| -> u32 {
            match v.partial_cmp(&0.0) {
                Some(core::cmp::Ordering::Greater) => round32(v.min(cell_cap) * SAT_SCALE) as u32,
                _ => 0,
            }
        };

        let mut cells_e = vec![0u32; n];
        let mut cells_c = vec![0u32; n];
        for i in 0..n {
            let e = sal_sq[i] + fg[i] + landmark_w[i] + classical[i];
            cells_e[i] = to_fixed(e);
            let (x, y) = (i % gw, i / gw);
            let xl = if x == 0 { fg[i] } else { fg[i - 1] };
            let xr = if x + 1 == gw { fg[i] } else { fg[i + 1] };
            let yt = if y == 0 { fg[i] } else { fg[i - gw] };
            let yb = if y + 1 == gh { fg[i] } else { fg[i + gw] };
            let grad = 0.5 * ((xr - xl).abs() + (yb - yt).abs());
            let hot = 5.0 * ((sal_sq[i] - 0.35) / 0.65).clamp(0.0, 1.0);
            cells_c[i] = to_fixed(cut[i] + hot + MATTE_GRADIENT_WEIGHT * grad);
        }

        let mut sem = vec![0.0f32; n];
        let mut max_sem = 0.0f32;
        for i in 0..n {
            let s = sal_sq[i].max(fg[i]);
            sem[i] = s;
            max_sem = max_sem.max(s);
        }
        let nbhd_mean = |x: usize, y: usize| -> f32 {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < gw && (ny as usize) < gh {
                        sum += sem[ny as usize * gw + nx as usize];
                    }
                }
            }
            sum / 9.0
        };
        let smoothed = |i: usize, thr: f32| -> bool {
            sem[i] >= thr && nbhd_mean(i % gw, i / gw) >= 0.65 * thr
        };
        let subject = if !faces.is_empty() || !focus_boxes.is_empty() {
            let (mut sx1, mut sy1, mut sx2, mut sy2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for f in &faces {
                let fw = f.fx2 - f.fx1;
                let fh = f.fy2 - f.fy1;
                sx1 = sx1.min(f.fx1 - 0.35 * fw);
                sy1 = sy1.min(f.headroom_y1);
                sx2 = sx2.max(f.fx2 + 0.35 * fw);
                sy2 = sy2.max(f.fy2 + 1.25 * fh);
            }
            for f in &focus_boxes {
                sx1 = sx1.min(f.fx1);
                sy1 = sy1.min(f.fy1);
                sx2 = sx2.max(f.fx2);
                sy2 = sy2.max(f.fy2);
            }
            if max_sem > 0.0 {
                let hi_thr = 0.50 * max_sem;
                let (ax1, ay1, ax2, ay2) = bbox_above(gw, gh, |i| smoothed(i, hi_thr));
                if ax2 > ax1 && ay2 > ay1 {
                    sx1 = sx1.min(ax1 as f32);
                    sy1 = sy1.min(ay1 as f32);
                    sx2 = sx2.max(ax2 as f32);
                    sy2 = sy2.max(ay2 as f32);
                }
            }
            let ix1 = (floor32(sx1) as i32).clamp(0, gw as i32) as usize;
            let iy1 = (floor32(sy1) as i32).clamp(0, gh as i32) as usize;
            let ix2 = (ceil32(sx2) as i32).clamp(0, gw as i32) as usize;
            let iy2 = (ceil32(sy2) as i32).clamp(0, gh as i32) as usize;
            (ix2 > ix1 && iy2 > iy1).then_some((ix1, iy1, ix2, iy2))
        } else {
            let t = self.subject_threshold * max_sem;
            let (sx1, sy1, sx2, sy2) = bbox_above(gw, gh, |i| smoothed(i, t));
            (sx2 > sx1 && sy2 > sy1).then_some((sx1, sy1, sx2, sy2))
        };

        let horizon_vec: Option<Vec<f32>> = self.horizon_profile.as_ref().map(|h| {
            if h.len() == gh {
                h.iter().copied().map(dequantize_u8).collect()
            } else if h.is_empty() {
                vec![0.0; gh]
            } else {
                (0..gh)
                    .map(|y| {
                        let fy = ((y as f32 + 0.5) * (h.len() as f32 / gh as f32) - 0.5)
                            .clamp(0.0, (h.len() - 1) as f32);
                        let y0 = floor32(fy) as usize;
                        let y1 = (y0 + 1).min(h.len() - 1);
                        let wy = fy - y0 as f32;
                        let v0 = dequantize_u8(h[y0]);
                        let v1 = dequantize_u8(h[y1]);
                        v0 + wy * (v1 - v0)
                    })
                    .collect()
            }
        });

        let map = DualIntegralMap::assemble(
            gw,
            gh,
            cells_e,
            cells_c,
            horizon_vec.as_deref(),
            subject,
            faces,
            focus_boxes,
        );
        (map, to_src)
    }

    /// Solve a single crop at `target_aspect` and `mode` for an image of
    /// dimensions `(src_w, src_h)` with optional runtime `manual_focus`.
    pub fn compute_crop(
        &self,
        src_w: u32,
        src_h: u32,
        target_aspect: (u32, u32),
        mode: SatCropMode,
        manual_focus: &[FocusRegion],
        solver_cfg: &SatCropConfig,
    ) -> Option<CropRect> {
        let (map, to_src) = self.build_integral_map(src_w, src_h, manual_focus);
        solve_crop_sat(&map, &to_src, target_aspect, mode, solver_cfg)
    }

    /// Solve multiple `(aspect, mode)` requests in a single pass (~15 µs SAT
    /// reconstruction + ~3 µs per aspect/mode candidate scan).
    pub fn compute_crops(
        &self,
        src_w: u32,
        src_h: u32,
        requests: &[((u32, u32), SatCropMode)],
        manual_focus: &[FocusRegion],
        solver_cfg: &SatCropConfig,
    ) -> Vec<Option<CropRect>> {
        let (map, to_src) = self.build_integral_map(src_w, src_h, manual_focus);
        requests
            .iter()
            .map(|&(aspect, mode)| solve_crop_sat(&map, &to_src, aspect, mode, solver_cfg))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Error returned when decoding [`CropMetadata`] from CBOR or JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataError {
    message: String,
}

impl MetadataError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }

    /// Human-readable error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetadataError {}

// ---------------------------------------------------------------------------
// Standard Base64 (RFC 4648) for JSON & Serde
// ---------------------------------------------------------------------------

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode a byte slice as standard RFC 4648 Base64.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let b0 = input[i] as u32;
        let b1 = input[i + 1] as u32;
        let b2 = input[i + 2] as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize]);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize]);
        out.push(B64_CHARS[((n >> 6) & 0x3f) as usize]);
        out.push(B64_CHARS[(n & 0x3f) as usize]);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize]);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize]);
        out.push(b'=');
        out.push(b'=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize]);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize]);
        out.push(B64_CHARS[((n >> 6) & 0x3f) as usize]);
        out.push(b'=');
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Decode a standard or URL-safe Base64 string into bytes.
pub fn base64_decode(input: &str) -> Result<Vec<u8>, MetadataError> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in input.as_bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let val = match b {
            b'A'..=b'Z' => (b - b'A') as u32,
            b'a'..=b'z' => (b - b'a' + 26) as u32,
            b'0'..=b'9' => (b - b'0' + 52) as u32,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(MetadataError::new("invalid base64 character")),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RFC 8949 CBOR Encoder & Decoder
// ---------------------------------------------------------------------------

fn cbor_write_type_len(buf: &mut Vec<u8>, major: u8, len: u64) {
    let mt = major << 5;
    if len < 24 {
        buf.push(mt | (len as u8));
    } else if len <= u8::MAX as u64 {
        buf.push(mt | 24);
        buf.push(len as u8);
    } else if len <= u16::MAX as u64 {
        buf.push(mt | 25);
        buf.extend_from_slice(&(len as u16).to_be_bytes());
    } else if len <= u32::MAX as u64 {
        buf.push(mt | 26);
        buf.extend_from_slice(&(len as u32).to_be_bytes());
    } else {
        buf.push(mt | 27);
        buf.extend_from_slice(&len.to_be_bytes());
    }
}

fn cbor_write_uint(buf: &mut Vec<u8>, val: u64) {
    cbor_write_type_len(buf, 0, val);
}

fn cbor_write_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    cbor_write_type_len(buf, 2, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn cbor_write_text(buf: &mut Vec<u8>, s: &str) {
    cbor_write_type_len(buf, 3, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn cbor_write_f32(buf: &mut Vec<u8>, val: f32) {
    buf.push(0xfa);
    buf.extend_from_slice(&val.to_be_bytes());
}

fn cbor_write_f32x4(buf: &mut Vec<u8>, arr: &[f32; 4]) {
    cbor_write_type_len(buf, 4, 4);
    for &v in arr {
        cbor_write_f32(buf, v);
    }
}

fn cbor_write_cue_plane(buf: &mut Vec<u8>, plane: &CuePlane) {
    cbor_write_type_len(buf, 5, 3);
    cbor_write_text(buf, "w");
    cbor_write_uint(buf, plane.width as u64);
    cbor_write_text(buf, "h");
    cbor_write_uint(buf, plane.height as u64);
    cbor_write_text(buf, "q8");
    cbor_write_bytes(buf, &plane.data);
}

fn cbor_write_subject(buf: &mut Vec<u8>, subj: &SubjectRegion) {
    let mut fields = 3u64; // kind, confidence, bbox
    if subj.protected_core.is_some() {
        fields += 1;
    }
    if subj.feature_band.is_some() {
        fields += 1;
    }
    if subj.preferred_margin.is_some() {
        fields += 1;
    }
    if !subj.keypoints.is_empty() {
        fields += 1;
    }
    cbor_write_type_len(buf, 5, fields);

    cbor_write_text(buf, "kind");
    cbor_write_text(buf, subj.kind.as_str());

    cbor_write_text(buf, "confidence");
    cbor_write_f32(buf, subj.confidence);

    cbor_write_text(buf, "bbox");
    cbor_write_f32x4(buf, &subj.bbox);

    if let Some(ref core) = subj.protected_core {
        cbor_write_text(buf, "core");
        cbor_write_f32x4(buf, core);
    }
    if let Some(ref band) = subj.feature_band {
        cbor_write_text(buf, "band");
        cbor_write_f32x4(buf, band);
    }
    if let Some(ref margin) = subj.preferred_margin {
        cbor_write_text(buf, "margin");
        cbor_write_f32x4(buf, margin);
    }
    if !subj.keypoints.is_empty() {
        cbor_write_text(buf, "keypoints");
        cbor_write_type_len(buf, 4, subj.keypoints.len() as u64);
        for kp in &subj.keypoints {
            cbor_write_type_len(buf, 4, 2);
            cbor_write_f32(buf, kp[0]);
            cbor_write_f32(buf, kp[1]);
        }
    }
}

/// Convert IEEE 754 half-precision (`binary16`) bits to `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let v = (mant as f32) * (1.0 / 16_777_216.0);
            if sign != 0 { -v } else { v }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

struct CborReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CborReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn peek(&self) -> Result<u8, MetadataError> {
        self.data
            .get(self.pos)
            .copied()
            .ok_or_else(|| MetadataError::new("unexpected EOF in CBOR"))
    }

    fn read_u8(&mut self) -> Result<u8, MetadataError> {
        let b = self.peek()?;
        self.pos += 1;
        Ok(b)
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], MetadataError> {
        if self.pos.saturating_add(n) > self.data.len() {
            return Err(MetadataError::new("unexpected EOF in CBOR payload"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_arg(&mut self, ai: u8) -> Result<u64, MetadataError> {
        match ai {
            0..=23 => Ok(ai as u64),
            24 => Ok(self.read_u8()? as u64),
            25 => {
                let b = self.read_exact(2)?;
                Ok(u16::from_be_bytes([b[0], b[1]]) as u64)
            }
            26 => {
                let b = self.read_exact(4)?;
                Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64)
            }
            27 => {
                let b = self.read_exact(8)?;
                Ok(u64::from_be_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
            _ => Err(MetadataError::new("invalid CBOR additional info")),
        }
    }

    fn skip_tags(&mut self) -> Result<(), MetadataError> {
        while self.pos < self.data.len() {
            let b = self.peek()?;
            if (b >> 5) == 6 {
                self.pos += 1;
                let _tag = self.read_arg(b & 0x1f)?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn read_uint(&mut self) -> Result<u64, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        let major = b >> 5;
        let ai = b & 0x1f;
        if major == 0 {
            self.read_arg(ai)
        } else {
            Err(MetadataError::new("expected CBOR unsigned integer"))
        }
    }

    fn read_f32(&mut self) -> Result<f32, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        let major = b >> 5;
        let ai = b & 0x1f;
        match (major, ai) {
            (0, _) => Ok(self.read_arg(ai)? as f32),
            (1, _) => Ok(-1.0 - (self.read_arg(ai)? as f32)),
            (7, 25) => {
                let bytes = self.read_exact(2)?;
                Ok(f16_to_f32(u16::from_be_bytes([bytes[0], bytes[1]])))
            }
            (7, 26) => {
                let bytes = self.read_exact(4)?;
                Ok(f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            (7, 27) => {
                let bytes = self.read_exact(8)?;
                Ok(f64::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as f32)
            }
            _ => Err(MetadataError::new("expected CBOR number")),
        }
    }

    fn read_text(&mut self) -> Result<String, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        let major = b >> 5;
        let ai = b & 0x1f;
        if major != 3 {
            return Err(MetadataError::new("expected CBOR text string"));
        }
        if ai == 31 {
            let mut out = String::new();
            while self.peek()? != 0xff {
                let chunk = self.read_text()?;
                out.push_str(&chunk);
            }
            self.pos += 1;
            return Ok(out);
        }
        let len = self.read_arg(ai)? as usize;
        let bytes = self.read_exact(len)?;
        core::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|_| MetadataError::new("invalid UTF-8 in CBOR text string"))
    }

    fn read_bytes_or_u8_seq(&mut self) -> Result<Vec<u8>, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        let major = b >> 5;
        let ai = b & 0x1f;
        match major {
            2 => {
                if ai == 31 {
                    let mut out = Vec::new();
                    while self.peek()? != 0xff {
                        let chunk = self.read_bytes_or_u8_seq()?;
                        out.extend_from_slice(&chunk);
                    }
                    self.pos += 1;
                    return Ok(out);
                }
                let len = self.read_arg(ai)? as usize;
                Ok(self.read_exact(len)?.to_vec())
            }
            3 => {
                let len = self.read_arg(ai)? as usize;
                let s = core::str::from_utf8(self.read_exact(len)?)
                    .map_err(|_| MetadataError::new("invalid UTF-8 in base64 field"))?;
                base64_decode(s)
            }
            4 => {
                if ai == 31 {
                    let mut out = Vec::new();
                    while self.peek()? != 0xff {
                        let v = self.read_uint()?;
                        if v > 255 {
                            return Err(MetadataError::new("u8 array element > 255"));
                        }
                        out.push(v as u8);
                    }
                    self.pos += 1;
                    return Ok(out);
                }
                let len = self.read_arg(ai)? as usize;
                let mut out = Vec::with_capacity(len.min(65536));
                for _ in 0..len {
                    let v = self.read_uint()?;
                    if v > 255 {
                        return Err(MetadataError::new("u8 array element > 255"));
                    }
                    out.push(v as u8);
                }
                Ok(out)
            }
            _ => Err(MetadataError::new(
                "expected CBOR byte string, u8 array, or base64 text",
            )),
        }
    }

    fn read_f32x4(&mut self) -> Result<[f32; 4], MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        if (b >> 5) != 4 {
            return Err(MetadataError::new("expected 4-element CBOR array"));
        }
        let ai = b & 0x1f;
        if ai == 31 {
            let v0 = self.read_f32()?;
            let v1 = self.read_f32()?;
            let v2 = self.read_f32()?;
            let v3 = self.read_f32()?;
            if self.read_u8()? != 0xff {
                return Err(MetadataError::new("expected 4-element CBOR array break"));
            }
            return Ok([v0, v1, v2, v3]);
        }
        let len = self.read_arg(ai)?;
        if len != 4 {
            return Err(MetadataError::new("expected 4-element CBOR array"));
        }
        Ok([
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
        ])
    }

    fn skip_value(&mut self, depth: u32) -> Result<(), MetadataError> {
        if depth > 32 {
            return Err(MetadataError::new("CBOR nesting depth exceeded"));
        }
        self.skip_tags()?;
        let b = self.read_u8()?;
        let major = b >> 5;
        let ai = b & 0x1f;
        match major {
            0 | 1 => {
                let _ = self.read_arg(ai)?;
                Ok(())
            }
            2 | 3 => {
                if ai == 31 {
                    while self.peek()? != 0xff {
                        self.skip_value(depth + 1)?;
                    }
                    self.pos += 1;
                    Ok(())
                } else {
                    let len = self.read_arg(ai)? as usize;
                    let _ = self.read_exact(len)?;
                    Ok(())
                }
            }
            4 => {
                if ai == 31 {
                    while self.peek()? != 0xff {
                        self.skip_value(depth + 1)?;
                    }
                    self.pos += 1;
                    Ok(())
                } else {
                    let len = self.read_arg(ai)?;
                    for _ in 0..len {
                        self.skip_value(depth + 1)?;
                    }
                    Ok(())
                }
            }
            5 => {
                if ai == 31 {
                    while self.peek()? != 0xff {
                        self.skip_value(depth + 1)?;
                        self.skip_value(depth + 1)?;
                    }
                    self.pos += 1;
                    Ok(())
                } else {
                    let len = self.read_arg(ai)?;
                    for _ in 0..len {
                        self.skip_value(depth + 1)?;
                        self.skip_value(depth + 1)?;
                    }
                    Ok(())
                }
            }
            7 => match ai {
                0..=23 => Ok(()),
                24 => {
                    let _ = self.read_u8()?;
                    Ok(())
                }
                25 => {
                    let _ = self.read_exact(2)?;
                    Ok(())
                }
                26 => {
                    let _ = self.read_exact(4)?;
                    Ok(())
                }
                27 => {
                    let _ = self.read_exact(8)?;
                    Ok(())
                }
                _ => Err(MetadataError::new("unsupported CBOR simple/float value")),
            },
            _ => Err(MetadataError::new("unsupported CBOR major type")),
        }
    }

    fn is_null(&mut self) -> Result<bool, MetadataError> {
        self.skip_tags()?;
        if self.peek()? == 0xf6 || self.peek()? == 0xf7 {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn read_map_header(&mut self) -> Result<Option<usize>, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        if (b >> 5) != 5 {
            return Err(MetadataError::new("expected CBOR map"));
        }
        let ai = b & 0x1f;
        if ai == 31 {
            Ok(None)
        } else {
            Ok(Some(self.read_arg(ai)? as usize))
        }
    }

    fn read_array_header(&mut self) -> Result<Option<usize>, MetadataError> {
        self.skip_tags()?;
        let b = self.read_u8()?;
        if (b >> 5) != 4 {
            return Err(MetadataError::new("expected CBOR array"));
        }
        let ai = b & 0x1f;
        if ai == 31 {
            Ok(None)
        } else {
            Ok(Some(self.read_arg(ai)? as usize))
        }
    }

    fn read_cue_plane(&mut self) -> Result<CuePlane, MetadataError> {
        let len_opt = self.read_map_header()?;
        let mut w: Option<u16> = None;
        let mut h: Option<u16> = None;
        let mut q8: Option<Vec<u8>> = None;

        let mut i = 0usize;
        loop {
            if let Some(n) = len_opt {
                if i >= n {
                    break;
                }
            } else if self.peek()? == 0xff {
                self.pos += 1;
                break;
            }
            i += 1;
            let key = self.read_text()?;
            match key.as_str() {
                "w" | "width" => {
                    let v = self.read_uint()?;
                    w = Some(
                        u16::try_from(v).map_err(|_| MetadataError::new("plane width > u16"))?,
                    );
                }
                "h" | "height" => {
                    let v = self.read_uint()?;
                    h = Some(
                        u16::try_from(v).map_err(|_| MetadataError::new("plane height > u16"))?,
                    );
                }
                "q8" | "q8_b64" | "data" => {
                    q8 = Some(self.read_bytes_or_u8_seq()?);
                }
                _ => self.skip_value(0)?,
            }
        }

        let width = w.ok_or_else(|| MetadataError::new("CuePlane missing 'w'"))?;
        let height = h.ok_or_else(|| MetadataError::new("CuePlane missing 'h'"))?;
        let data = q8.ok_or_else(|| MetadataError::new("CuePlane missing 'q8'"))?;
        if data.len() != width as usize * height as usize {
            return Err(MetadataError::new("CuePlane data length mismatch"));
        }
        Ok(CuePlane {
            width,
            height,
            data,
        })
    }

    fn read_subject(&mut self) -> Result<SubjectRegion, MetadataError> {
        let len_opt = self.read_map_header()?;
        let mut kind = SubjectKind::Face;
        let mut confidence = 1.0f32;
        let mut bbox: Option<[f32; 4]> = None;
        let mut protected_core: Option<[f32; 4]> = None;
        let mut feature_band: Option<[f32; 4]> = None;
        let mut preferred_margin: Option<[f32; 4]> = None;
        let mut keypoints: Vec<[f32; 2]> = Vec::new();

        let mut i = 0usize;
        loop {
            if let Some(n) = len_opt {
                if i >= n {
                    break;
                }
            } else if self.peek()? == 0xff {
                self.pos += 1;
                break;
            }
            i += 1;
            let key = self.read_text()?;
            match key.as_str() {
                "kind" => {
                    let s = self.read_text()?;
                    kind = SubjectKind::from_str_tag(&s);
                }
                "confidence" => {
                    confidence = self.read_f32()?;
                }
                "bbox" => {
                    bbox = Some(self.read_f32x4()?);
                }
                "core" | "protected_core" => {
                    if !self.is_null()? {
                        protected_core = Some(self.read_f32x4()?);
                    }
                }
                "band" | "feature_band" => {
                    if !self.is_null()? {
                        feature_band = Some(self.read_f32x4()?);
                    }
                }
                "margin" | "preferred_margin" => {
                    if !self.is_null()? {
                        preferred_margin = Some(self.read_f32x4()?);
                    }
                }
                "keypoints" => {
                    if !self.is_null()? {
                        let arr_len = self.read_array_header()?;
                        let mut k = 0usize;
                        loop {
                            if let Some(an) = arr_len {
                                if k >= an {
                                    break;
                                }
                            } else if self.peek()? == 0xff {
                                self.pos += 1;
                                break;
                            }
                            k += 1;
                            let pt_len = self.read_array_header()?;
                            let x = self.read_f32()?;
                            let y = self.read_f32()?;
                            if let Some(pn) = pt_len {
                                for _ in 2..pn {
                                    self.skip_value(0)?;
                                }
                            } else if self.read_u8()? != 0xff {
                                return Err(MetadataError::new("expected keypoint break"));
                            }
                            keypoints.push([x, y]);
                        }
                    }
                }
                _ => self.skip_value(0)?,
            }
        }

        let bbox = bbox.ok_or_else(|| MetadataError::new("SubjectRegion missing 'bbox'"))?;
        Ok(SubjectRegion {
            kind,
            confidence,
            bbox,
            protected_core,
            feature_band,
            preferred_margin,
            keypoints,
        })
    }
}

impl CropMetadata {
    /// Encode this [`CropMetadata`] as a standard RFC 8949 CBOR map (`Major Type 5`)
    /// with string keys and native byte strings (`Major Type 2`) for 2D [`CuePlane`]s.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut fields = 3u64; // v, subject_threshold, subjects
        if self.producer.is_some() {
            fields += 1;
        }
        if self.aspect_ratio.is_some() {
            fields += 1;
        }
        if self.saliency.is_some() {
            fields += 1;
        }
        if self.foreground.is_some() {
            fields += 1;
        }
        if self.structural_energy.is_some() {
            fields += 1;
        }
        if self.horizon_profile.is_some() {
            fields += 1;
        }

        let mut buf = Vec::with_capacity(256);
        cbor_write_type_len(&mut buf, 5, fields);

        cbor_write_text(&mut buf, "v");
        cbor_write_uint(&mut buf, self.version as u64);

        if let Some(ref prod) = self.producer {
            cbor_write_text(&mut buf, "producer");
            cbor_write_text(&mut buf, prod);
        }
        if let Some(aspect) = self.aspect_ratio {
            cbor_write_text(&mut buf, "aspect");
            cbor_write_f32(&mut buf, aspect);
        }

        cbor_write_text(&mut buf, "subject_threshold");
        cbor_write_f32(&mut buf, self.subject_threshold);

        cbor_write_text(&mut buf, "subjects");
        cbor_write_type_len(&mut buf, 4, self.subjects.len() as u64);
        for subj in &self.subjects {
            cbor_write_subject(&mut buf, subj);
        }

        if let Some(ref p) = self.saliency {
            cbor_write_text(&mut buf, "saliency");
            cbor_write_cue_plane(&mut buf, p);
        }
        if let Some(ref p) = self.foreground {
            cbor_write_text(&mut buf, "foreground");
            cbor_write_cue_plane(&mut buf, p);
        }
        if let Some(ref p) = self.structural_energy {
            cbor_write_text(&mut buf, "structural_energy");
            cbor_write_cue_plane(&mut buf, p);
        }
        if let Some(ref h) = self.horizon_profile {
            cbor_write_text(&mut buf, "horizon_profile");
            cbor_write_bytes(&mut buf, h);
        }

        buf
    }

    /// Decode [`CropMetadata`] from a standard RFC 8949 CBOR byte slice.
    ///
    /// Ignores unknown fields of any CBOR major type for forward compatibility
    /// and accepts optional CBOR tags (such as tag `55799`).
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MetadataError> {
        let mut r = CborReader::new(bytes);
        let len_opt = r.read_map_header()?;
        let mut meta = Self::default();

        let mut i = 0usize;
        loop {
            if let Some(n) = len_opt {
                if i >= n {
                    break;
                }
            } else if r.peek()? == 0xff {
                break;
            }
            i += 1;
            let key = r.read_text()?;
            match key.as_str() {
                "v" | "version" => {
                    meta.version = r.read_uint()? as u32;
                }
                "producer" => {
                    if !r.is_null()? {
                        meta.producer = Some(r.read_text()?);
                    }
                }
                "aspect" | "aspect_ratio" => {
                    if !r.is_null()? {
                        meta.aspect_ratio = Some(r.read_f32()?);
                    }
                }
                "subject_threshold" => {
                    meta.subject_threshold = r.read_f32()?;
                }
                "subjects" => {
                    if !r.is_null()? {
                        let arr_len = r.read_array_header()?;
                        let mut k = 0usize;
                        loop {
                            if let Some(an) = arr_len {
                                if k >= an {
                                    break;
                                }
                            } else if r.peek()? == 0xff {
                                r.pos += 1;
                                break;
                            }
                            k += 1;
                            meta.subjects.push(r.read_subject()?);
                        }
                    }
                }
                "saliency" => {
                    if !r.is_null()? {
                        meta.saliency = Some(r.read_cue_plane()?);
                    }
                }
                "foreground" => {
                    if !r.is_null()? {
                        meta.foreground = Some(r.read_cue_plane()?);
                    }
                }
                "structural_energy" => {
                    if !r.is_null()? {
                        meta.structural_energy = Some(r.read_cue_plane()?);
                    }
                }
                "horizon_profile" | "horizon_profile_b64" => {
                    if !r.is_null()? {
                        meta.horizon_profile = Some(r.read_bytes_or_u8_seq()?);
                    }
                }
                _ => r.skip_value(0)?,
            }
        }
        Ok(meta)
    }
}

// ---------------------------------------------------------------------------
// RFC 8259 JSON Encoder & Decoder (Zero-Dependency)
// ---------------------------------------------------------------------------

fn json_escape_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use core::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_write_f32(out: &mut String, v: f32) {
    use core::fmt::Write;
    if v.is_finite() {
        let _ = write!(out, "{}", v);
    } else {
        out.push_str("0.0");
    }
}

fn json_write_f32x4(out: &mut String, arr: &[f32; 4]) {
    out.push('[');
    for (i, &v) in arr.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        json_write_f32(out, v);
    }
    out.push(']');
}

fn json_write_cue_plane(out: &mut String, p: &CuePlane) {
    use core::fmt::Write;
    let _ = write!(out, "{{\"w\":{},\"h\":{},\"q8_b64\":\"", p.width, p.height);
    out.push_str(&base64_encode(&p.data));
    out.push_str("\"}");
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Result<u8, MetadataError> {
        self.skip_ws();
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| MetadataError::new("unexpected EOF in JSON"))
    }

    fn expect_char(&mut self, expected: u8) -> Result<(), MetadataError> {
        let b = self.peek()?;
        if b == expected {
            self.pos += 1;
            Ok(())
        } else {
            Err(MetadataError::new("unexpected token in JSON"))
        }
    }

    fn consume_null(&mut self) -> Result<bool, MetadataError> {
        self.skip_ws();
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn read_string(&mut self) -> Result<String, MetadataError> {
        self.expect_char(b'"')?;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                if self.pos >= self.bytes.len() {
                    return Err(MetadataError::new("unterminated escape in JSON string"));
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err(MetadataError::new("short \\u escape in JSON"));
                        }
                        let hex = core::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| MetadataError::new("invalid \\u hex"))?;
                        self.pos += 4;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| MetadataError::new("invalid \\u hex"))?;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                    _ => return Err(MetadataError::new("invalid escape in JSON string")),
                }
            } else {
                // Fast path for ASCII or multi-byte UTF-8 sequence.
                self.pos -= 1;
                let rest = core::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| MetadataError::new("invalid UTF-8 in JSON string"))?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| MetadataError::new("unexpected EOF"))?;
                self.pos += ch.len_utf8();
                out.push(ch);
            }
        }
        Err(MetadataError::new("unterminated JSON string"))
    }

    fn read_number_str(&mut self) -> Result<&'a str, MetadataError> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(MetadataError::new("expected JSON number"));
        }
        core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| MetadataError::new("invalid JSON number"))
    }

    fn read_f32(&mut self) -> Result<f32, MetadataError> {
        let s = self.read_number_str()?;
        s.parse::<f32>()
            .map_err(|_| MetadataError::new("invalid float in JSON"))
    }

    fn read_u64(&mut self) -> Result<u64, MetadataError> {
        let s = self.read_number_str()?;
        s.parse::<u64>()
            .map_err(|_| MetadataError::new("invalid unsigned integer in JSON"))
    }

    fn read_f32x4(&mut self) -> Result<[f32; 4], MetadataError> {
        self.expect_char(b'[')?;
        let v0 = self.read_f32()?;
        self.expect_char(b',')?;
        let v1 = self.read_f32()?;
        self.expect_char(b',')?;
        let v2 = self.read_f32()?;
        self.expect_char(b',')?;
        let v3 = self.read_f32()?;
        self.expect_char(b']')?;
        Ok([v0, v1, v2, v3])
    }

    fn read_bytes_or_u8_array(&mut self) -> Result<Vec<u8>, MetadataError> {
        let b = self.peek()?;
        if b == b'"' {
            let s = self.read_string()?;
            base64_decode(&s)
        } else if b == b'[' {
            self.pos += 1;
            let mut out = Vec::new();
            if self.peek()? == b']' {
                self.pos += 1;
                return Ok(out);
            }
            loop {
                let v = self.read_u64()?;
                if v > 255 {
                    return Err(MetadataError::new("JSON u8 array value > 255"));
                }
                out.push(v as u8);
                let sep = self.peek()?;
                if sep == b',' {
                    self.pos += 1;
                } else if sep == b']' {
                    self.pos += 1;
                    break;
                } else {
                    return Err(MetadataError::new("expected ',' or ']' in JSON array"));
                }
            }
            Ok(out)
        } else {
            Err(MetadataError::new("expected base64 string or u8 array"))
        }
    }

    fn skip_value(&mut self, depth: u32) -> Result<(), MetadataError> {
        if depth > 32 {
            return Err(MetadataError::new("JSON nesting depth exceeded"));
        }
        let b = self.peek()?;
        match b {
            b'"' => {
                let _ = self.read_string()?;
                Ok(())
            }
            b'{' => {
                self.pos += 1;
                if self.peek()? == b'}' {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    let _ = self.read_string()?;
                    self.expect_char(b':')?;
                    self.skip_value(depth + 1)?;
                    let sep = self.peek()?;
                    if sep == b',' {
                        self.pos += 1;
                    } else if sep == b'}' {
                        self.pos += 1;
                        break;
                    } else {
                        return Err(MetadataError::new("expected ',' or '}' in JSON object"));
                    }
                }
                Ok(())
            }
            b'[' => {
                self.pos += 1;
                if self.peek()? == b']' {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value(depth + 1)?;
                    let sep = self.peek()?;
                    if sep == b',' {
                        self.pos += 1;
                    } else if sep == b']' {
                        self.pos += 1;
                        break;
                    } else {
                        return Err(MetadataError::new("expected ',' or ']' in JSON array"));
                    }
                }
                Ok(())
            }
            b't' if self.bytes[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(())
            }
            b'f' if self.bytes[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(())
            }
            b'n' if self.bytes[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Ok(())
            }
            _ => {
                let _ = self.read_number_str()?;
                Ok(())
            }
        }
    }

    fn read_cue_plane(&mut self) -> Result<CuePlane, MetadataError> {
        self.expect_char(b'{')?;
        let mut w: Option<u16> = None;
        let mut h: Option<u16> = None;
        let mut data: Option<Vec<u8>> = None;

        if self.peek()? != b'}' {
            loop {
                let key = self.read_string()?;
                self.expect_char(b':')?;
                match key.as_str() {
                    "w" | "width" => {
                        let v = self.read_u64()?;
                        w = Some(
                            u16::try_from(v)
                                .map_err(|_| MetadataError::new("plane width > u16"))?,
                        );
                    }
                    "h" | "height" => {
                        let v = self.read_u64()?;
                        h = Some(
                            u16::try_from(v)
                                .map_err(|_| MetadataError::new("plane height > u16"))?,
                        );
                    }
                    "q8_b64" | "q8" | "data" => {
                        data = Some(self.read_bytes_or_u8_array()?);
                    }
                    _ => self.skip_value(0)?,
                }
                let sep = self.peek()?;
                if sep == b',' {
                    self.pos += 1;
                } else if sep == b'}' {
                    break;
                } else {
                    return Err(MetadataError::new("expected ',' or '}' in CuePlane JSON"));
                }
            }
        }
        self.expect_char(b'}')?;

        let width = w.ok_or_else(|| MetadataError::new("CuePlane missing 'w'"))?;
        let height = h.ok_or_else(|| MetadataError::new("CuePlane missing 'h'"))?;
        let data = data.ok_or_else(|| MetadataError::new("CuePlane missing 'q8_b64' or 'q8'"))?;
        if data.len() != width as usize * height as usize {
            return Err(MetadataError::new("CuePlane data length mismatch"));
        }
        Ok(CuePlane {
            width,
            height,
            data,
        })
    }

    fn read_subject(&mut self) -> Result<SubjectRegion, MetadataError> {
        self.expect_char(b'{')?;
        let mut kind = SubjectKind::Face;
        let mut confidence = 1.0f32;
        let mut bbox: Option<[f32; 4]> = None;
        let mut protected_core: Option<[f32; 4]> = None;
        let mut feature_band: Option<[f32; 4]> = None;
        let mut preferred_margin: Option<[f32; 4]> = None;
        let mut keypoints: Vec<[f32; 2]> = Vec::new();

        if self.peek()? != b'}' {
            loop {
                let key = self.read_string()?;
                self.expect_char(b':')?;
                match key.as_str() {
                    "kind" => {
                        let s = self.read_string()?;
                        kind = SubjectKind::from_str_tag(&s);
                    }
                    "confidence" => {
                        confidence = self.read_f32()?;
                    }
                    "bbox" => {
                        bbox = Some(self.read_f32x4()?);
                    }
                    "core" | "protected_core" => {
                        if !self.consume_null()? {
                            protected_core = Some(self.read_f32x4()?);
                        }
                    }
                    "band" | "feature_band" => {
                        if !self.consume_null()? {
                            feature_band = Some(self.read_f32x4()?);
                        }
                    }
                    "margin" | "preferred_margin" => {
                        if !self.consume_null()? {
                            preferred_margin = Some(self.read_f32x4()?);
                        }
                    }
                    "keypoints" => {
                        if !self.consume_null()? {
                            self.expect_char(b'[')?;
                            if self.peek()? != b']' {
                                loop {
                                    self.expect_char(b'[')?;
                                    let x = self.read_f32()?;
                                    self.expect_char(b',')?;
                                    let y = self.read_f32()?;
                                    self.expect_char(b']')?;
                                    keypoints.push([x, y]);
                                    let sep = self.peek()?;
                                    if sep == b',' {
                                        self.pos += 1;
                                    } else if sep == b']' {
                                        break;
                                    } else {
                                        return Err(MetadataError::new(
                                            "expected ',' or ']' in keypoints",
                                        ));
                                    }
                                }
                            }
                            self.expect_char(b']')?;
                        }
                    }
                    _ => self.skip_value(0)?,
                }
                let sep = self.peek()?;
                if sep == b',' {
                    self.pos += 1;
                } else if sep == b'}' {
                    break;
                } else {
                    return Err(MetadataError::new(
                        "expected ',' or '}' in SubjectRegion JSON",
                    ));
                }
            }
        }
        self.expect_char(b'}')?;

        let bbox = bbox.ok_or_else(|| MetadataError::new("SubjectRegion missing 'bbox'"))?;
        Ok(SubjectRegion {
            kind,
            confidence,
            bbox,
            protected_core,
            feature_band,
            preferred_margin,
            keypoints,
        })
    }
}

impl CropMetadata {
    /// Encode this [`CropMetadata`] as a compact RFC 8259 JSON string with Base64
    /// (`"q8_b64"`) [`CuePlane`] payloads.
    pub fn to_json(&self) -> String {
        use core::fmt::Write;
        let mut out = String::with_capacity(512);
        let _ = write!(out, "{{\"v\":{}", self.version);

        if let Some(ref prod) = self.producer {
            out.push_str(",\"producer\":");
            json_escape_str(&mut out, prod);
        }
        if let Some(aspect) = self.aspect_ratio {
            out.push_str(",\"aspect\":");
            json_write_f32(&mut out, aspect);
        }
        out.push_str(",\"subject_threshold\":");
        json_write_f32(&mut out, self.subject_threshold);

        out.push_str(",\"subjects\":[");
        for (i, subj) in self.subjects.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"kind\":");
            json_escape_str(&mut out, subj.kind.as_str());
            out.push_str(",\"confidence\":");
            json_write_f32(&mut out, subj.confidence);
            out.push_str(",\"bbox\":");
            json_write_f32x4(&mut out, &subj.bbox);
            if let Some(ref core) = subj.protected_core {
                out.push_str(",\"core\":");
                json_write_f32x4(&mut out, core);
            }
            if let Some(ref band) = subj.feature_band {
                out.push_str(",\"band\":");
                json_write_f32x4(&mut out, band);
            }
            if let Some(ref margin) = subj.preferred_margin {
                out.push_str(",\"margin\":");
                json_write_f32x4(&mut out, margin);
            }
            if !subj.keypoints.is_empty() {
                out.push_str(",\"keypoints\":[");
                for (ki, kp) in subj.keypoints.iter().enumerate() {
                    if ki > 0 {
                        out.push(',');
                    }
                    out.push('[');
                    json_write_f32(&mut out, kp[0]);
                    out.push(',');
                    json_write_f32(&mut out, kp[1]);
                    out.push(']');
                }
                out.push(']');
            }
            out.push('}');
        }
        out.push(']');

        if let Some(ref p) = self.saliency {
            out.push_str(",\"saliency\":");
            json_write_cue_plane(&mut out, p);
        }
        if let Some(ref p) = self.foreground {
            out.push_str(",\"foreground\":");
            json_write_cue_plane(&mut out, p);
        }
        if let Some(ref p) = self.structural_energy {
            out.push_str(",\"structural_energy\":");
            json_write_cue_plane(&mut out, p);
        }
        if let Some(ref h) = self.horizon_profile {
            out.push_str(",\"horizon_profile_b64\":\"");
            out.push_str(&base64_encode(h));
            out.push('"');
        }
        out.push('}');
        out
    }

    /// Decode [`CropMetadata`] from a JSON string.
    ///
    /// Ignores unknown fields for forward compatibility and accepts both Base64
    /// (`"q8_b64"`) and integer array (`"q8": [0..=255]`) representations for planes.
    pub fn from_json(json: &str) -> Result<Self, MetadataError> {
        let mut p = JsonParser::new(json);
        p.expect_char(b'{')?;
        let mut meta = Self::default();

        if p.peek()? != b'}' {
            loop {
                let key = p.read_string()?;
                p.expect_char(b':')?;
                match key.as_str() {
                    "v" | "version" => {
                        meta.version = p.read_u64()? as u32;
                    }
                    "producer" => {
                        if !p.consume_null()? {
                            meta.producer = Some(p.read_string()?);
                        }
                    }
                    "aspect" | "aspect_ratio" => {
                        if !p.consume_null()? {
                            meta.aspect_ratio = Some(p.read_f32()?);
                        }
                    }
                    "subject_threshold" => {
                        meta.subject_threshold = p.read_f32()?;
                    }
                    "subjects" => {
                        if !p.consume_null()? {
                            p.expect_char(b'[')?;
                            if p.peek()? != b']' {
                                loop {
                                    meta.subjects.push(p.read_subject()?);
                                    let sep = p.peek()?;
                                    if sep == b',' {
                                        p.pos += 1;
                                    } else if sep == b']' {
                                        break;
                                    } else {
                                        return Err(MetadataError::new(
                                            "expected ',' or ']' in subjects array",
                                        ));
                                    }
                                }
                            }
                            p.expect_char(b']')?;
                        }
                    }
                    "saliency" => {
                        if !p.consume_null()? {
                            meta.saliency = Some(p.read_cue_plane()?);
                        }
                    }
                    "foreground" => {
                        if !p.consume_null()? {
                            meta.foreground = Some(p.read_cue_plane()?);
                        }
                    }
                    "structural_energy" => {
                        if !p.consume_null()? {
                            meta.structural_energy = Some(p.read_cue_plane()?);
                        }
                    }
                    "horizon_profile" | "horizon_profile_b64" => {
                        if !p.consume_null()? {
                            meta.horizon_profile = Some(p.read_bytes_or_u8_array()?);
                        }
                    }
                    _ => p.skip_value(0)?,
                }
                let sep = p.peek()?;
                if sep == b',' {
                    p.pos += 1;
                } else if sep == b'}' {
                    break;
                } else {
                    return Err(MetadataError::new(
                        "expected ',' or '}' in CropMetadata JSON",
                    ));
                }
            }
        }
        p.expect_char(b'}')?;
        Ok(meta)
    }
}

// ---------------------------------------------------------------------------
// Optional `serde` Support
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use serde::de::{self, MapAccess, SeqAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for SubjectKind {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_str(self.as_str())
        }
    }

    impl<'de> Deserialize<'de> for SubjectKind {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            Ok(Self::from_str_tag(&s))
        }
    }

    impl Serialize for CuePlane {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let human_readable = serializer.is_human_readable();
            let mut st = serializer.serialize_struct("CuePlane", 3)?;
            st.serialize_field("w", &self.width)?;
            st.serialize_field("h", &self.height)?;
            if human_readable {
                let b64 = base64_encode(&self.data);
                st.serialize_field("q8_b64", &b64)?;
            } else {
                st.serialize_field("q8", &self.data)?;
            }
            st.end()
        }
    }

    enum BytesOrB64 {
        Bytes(Vec<u8>),
    }

    impl<'de> Deserialize<'de> for BytesOrB64 {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct BytesOrB64Visitor;
            impl<'de> Visitor<'de> for BytesOrB64Visitor {
                type Value = BytesOrB64;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a base64 string, byte slice, or u8 sequence")
                }
                fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                    base64_decode(v)
                        .map(BytesOrB64::Bytes)
                        .map_err(|e| E::custom(e.message()))
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                    Ok(BytesOrB64::Bytes(v.to_vec()))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                    Ok(BytesOrB64::Bytes(v))
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                    while let Some(b) = seq.next_element::<u8>()? {
                        out.push(b);
                    }
                    Ok(BytesOrB64::Bytes(out))
                }
            }
            deserializer.deserialize_any(BytesOrB64Visitor)
        }
    }

    impl<'de> Deserialize<'de> for CuePlane {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct CuePlaneVisitor;
            impl<'de> Visitor<'de> for CuePlaneVisitor {
                type Value = CuePlane;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a CuePlane map with w, h, and q8 or q8_b64")
                }
                fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                    let mut w: Option<u16> = None;
                    let mut h: Option<u16> = None;
                    let mut data: Option<Vec<u8>> = None;
                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "w" | "width" => w = Some(map.next_value()?),
                            "h" | "height" => h = Some(map.next_value()?),
                            "q8" | "q8_b64" | "data" => {
                                let BytesOrB64::Bytes(b) = map.next_value::<BytesOrB64>()?;
                                data = Some(b);
                            }
                            _ => {
                                let _ = map.next_value::<de::IgnoredAny>()?;
                            }
                        }
                    }
                    let width = w.ok_or_else(|| de::Error::missing_field("w"))?;
                    let height = h.ok_or_else(|| de::Error::missing_field("h"))?;
                    let data = data.ok_or_else(|| de::Error::missing_field("q8_b64"))?;
                    if data.len() != width as usize * height as usize {
                        return Err(de::Error::custom("CuePlane data length mismatch"));
                    }
                    Ok(CuePlane {
                        width,
                        height,
                        data,
                    })
                }
            }
            deserializer.deserialize_map(CuePlaneVisitor)
        }
    }

    impl Serialize for SubjectRegion {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut st = serializer.serialize_struct("SubjectRegion", 7)?;
            st.serialize_field("kind", &self.kind)?;
            st.serialize_field("confidence", &self.confidence)?;
            st.serialize_field("bbox", &self.bbox)?;
            if let Some(ref core) = self.protected_core {
                st.serialize_field("core", core)?;
            }
            if let Some(ref band) = self.feature_band {
                st.serialize_field("band", band)?;
            }
            if let Some(ref margin) = self.preferred_margin {
                st.serialize_field("margin", margin)?;
            }
            if !self.keypoints.is_empty() {
                st.serialize_field("keypoints", &self.keypoints)?;
            }
            st.end()
        }
    }

    impl<'de> Deserialize<'de> for SubjectRegion {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct SubjectVisitor;
            impl<'de> Visitor<'de> for SubjectVisitor {
                type Value = SubjectRegion;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a SubjectRegion map")
                }
                fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                    let mut kind = SubjectKind::Face;
                    let mut confidence = 1.0f32;
                    let mut bbox: Option<[f32; 4]> = None;
                    let mut protected_core: Option<[f32; 4]> = None;
                    let mut feature_band: Option<[f32; 4]> = None;
                    let mut preferred_margin: Option<[f32; 4]> = None;
                    let mut keypoints: Vec<[f32; 2]> = Vec::new();

                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "kind" => kind = map.next_value()?,
                            "confidence" => confidence = map.next_value()?,
                            "bbox" => bbox = Some(map.next_value()?),
                            "core" | "protected_core" => protected_core = map.next_value()?,
                            "band" | "feature_band" => feature_band = map.next_value()?,
                            "margin" | "preferred_margin" => preferred_margin = map.next_value()?,
                            "keypoints" => {
                                let kps: Option<Vec<[f32; 2]>> = map.next_value()?;
                                keypoints = kps.unwrap_or_default();
                            }
                            _ => {
                                let _ = map.next_value::<de::IgnoredAny>()?;
                            }
                        }
                    }
                    let bbox = bbox.ok_or_else(|| de::Error::missing_field("bbox"))?;
                    Ok(SubjectRegion {
                        kind,
                        confidence,
                        bbox,
                        protected_core,
                        feature_band,
                        preferred_margin,
                        keypoints,
                    })
                }
            }
            deserializer.deserialize_map(SubjectVisitor)
        }
    }

    impl Serialize for CropMetadata {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let human_readable = serializer.is_human_readable();
            let mut st = serializer.serialize_struct("CropMetadata", 9)?;
            st.serialize_field("v", &self.version)?;
            if let Some(ref prod) = self.producer {
                st.serialize_field("producer", prod)?;
            }
            if let Some(ref aspect) = self.aspect_ratio {
                st.serialize_field("aspect", aspect)?;
            }
            st.serialize_field("subject_threshold", &self.subject_threshold)?;
            st.serialize_field("subjects", &self.subjects)?;
            if let Some(ref p) = self.saliency {
                st.serialize_field("saliency", p)?;
            }
            if let Some(ref p) = self.foreground {
                st.serialize_field("foreground", p)?;
            }
            if let Some(ref p) = self.structural_energy {
                st.serialize_field("structural_energy", p)?;
            }
            if let Some(ref h) = self.horizon_profile {
                if human_readable {
                    let b64 = base64_encode(h);
                    st.serialize_field("horizon_profile_b64", &b64)?;
                } else {
                    st.serialize_field("horizon_profile", h)?;
                }
            }
            st.end()
        }
    }

    impl<'de> Deserialize<'de> for CropMetadata {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct MetaVisitor;
            impl<'de> Visitor<'de> for MetaVisitor {
                type Value = CropMetadata;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a CropMetadata map")
                }
                fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                    let mut meta = CropMetadata::default();
                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "v" | "version" => meta.version = map.next_value()?,
                            "producer" => meta.producer = map.next_value()?,
                            "aspect" | "aspect_ratio" => meta.aspect_ratio = map.next_value()?,
                            "subject_threshold" => meta.subject_threshold = map.next_value()?,
                            "subjects" => {
                                let subjs: Option<Vec<SubjectRegion>> = map.next_value()?;
                                meta.subjects = subjs.unwrap_or_default();
                            }
                            "saliency" => meta.saliency = map.next_value()?,
                            "foreground" => meta.foreground = map.next_value()?,
                            "structural_energy" => meta.structural_energy = map.next_value()?,
                            "horizon_profile" | "horizon_profile_b64" => {
                                let opt: Option<BytesOrB64> = map.next_value()?;
                                meta.horizon_profile = opt.map(|BytesOrB64::Bytes(b)| b);
                            }
                            _ => {
                                let _ = map.next_value::<de::IgnoredAny>()?;
                            }
                        }
                    }
                    Ok(meta)
                }
            }
            deserializer.deserialize_map(MetaVisitor)
        }
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaceRect, SaliencyMap};

    fn sample_metadata() -> CropMetadata {
        let gw = 64usize;
        let gh = 48usize;
        let mut sal_data = vec![0.0f32; 128 * 128];
        for y in 32..96 {
            for x in 40..88 {
                sal_data[y * 128 + x] = 0.85;
            }
        }
        let sal = SaliencyMap {
            data: sal_data,
            width: 128,
            height: 128,
        };
        let mut classical = vec![0.05f32; gw * gh];
        for y in 12..36 {
            for x in 18..44 {
                classical[y * gw + x] = 0.45;
            }
        }
        let horizon: Vec<f32> = (0..gh).map(|y| if y == 16 { 0.8 } else { 0.1 }).collect();
        let faces = [FaceLandmarkInput {
            rect: FaceRect {
                x1: 35.0,
                y1: 20.0,
                x2: 55.0,
                y2: 48.0,
                confidence: 0.94,
            },
            landmarks: Some([
                [40.0, 29.0],
                [50.0, 29.0],
                [45.0, 35.0],
                [41.0, 41.0],
                [49.0, 41.0],
            ]),
        }];
        let to_src = TargetToSrc::stretch(1200, 900, 512, 384);
        CropMetadata::from_crop_inputs(
            &CropInputSources {
                saliency: Some(&sal),
                matte: None,
                faces: &faces,
                manual_focus: &[],
                classical_energy: Some(&classical),
                horizon_profile: Some(&horizon),
            },
            gw,
            gh,
            &to_src,
            0.25,
            Some("zensally/0.2.0"),
        )
    }

    #[test]
    fn cbor_roundtrip_and_self_describe_tag() {
        let meta = sample_metadata();
        let cbor = meta.to_cbor();
        assert!(
            cbor.len() < 10_000,
            "64x48 3-plane CBOR size {} >= 10 KB",
            cbor.len()
        );
        let decoded = CropMetadata::from_cbor(&cbor).expect("valid CBOR");
        assert_eq!(decoded, meta);

        // Wrap in RFC 8949 self-described CBOR tag 55799 (0xd9d9f7)
        let mut tagged = vec![0xd9, 0xd9, 0xf7];
        tagged.extend_from_slice(&cbor);
        let decoded_tagged = CropMetadata::from_cbor(&tagged).expect("tagged CBOR");
        assert_eq!(decoded_tagged, meta);
    }

    #[test]
    fn json_roundtrip_and_unknown_fields() {
        let meta = sample_metadata();
        let json = meta.to_json();
        let decoded = CropMetadata::from_json(&json).expect("valid JSON");
        assert_eq!(decoded.version, meta.version);
        assert_eq!(decoded.producer, meta.producer);
        assert_eq!(decoded.saliency, meta.saliency);
        assert_eq!(decoded.foreground, meta.foreground);
        assert_eq!(decoded.structural_energy, meta.structural_energy);
        assert_eq!(decoded.horizon_profile, meta.horizon_profile);
        assert_eq!(decoded.subjects.len(), 1);
        assert_eq!(decoded.subjects[0].kind, SubjectKind::Face);

        // Verify forward compatibility with unknown JSON fields and custom subject kind.
        let forward_json = r#"{
            "v": 2,
            "producer": "future-ai/3.0",
            "future_model_meta": {"backbone": "vit-nano", "layers": [1, 2, 3]},
            "subjects": [
                {
                    "kind": "bird",
                    "confidence": 0.88,
                    "bbox": [0.2, 0.3, 0.6, 0.7],
                    "extra_attr": "perched"
                }
            ],
            "saliency": {"w": 2, "h": 2, "q8": [0, 64, 128, 255], "extra": true}
        }"#;
        let parsed = CropMetadata::from_json(forward_json).expect("forward-compatible JSON");
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.subjects[0].kind, SubjectKind::Custom("bird".into()));
        assert_eq!(
            parsed.saliency.as_ref().unwrap().data,
            vec![0, 64, 128, 255]
        );
    }

    #[test]
    fn compact_32x24_and_crop_derivation_parity() {
        let meta = sample_metadata();
        let cfg = SatCropConfig::default();
        let reqs = [
            ((1, 1), SatCropMode::Minimal),
            ((1, 1), SatCropMode::Maximal),
            ((16, 9), SatCropMode::Minimal),
            ((9, 16), SatCropMode::Maximal),
        ];

        let crops_full = meta.compute_crops(1200, 900, &reqs, &[], &cfg);
        let compact = meta.clone().into_compact_32x24();
        let cbor_compact = compact.to_cbor();
        assert!(
            cbor_compact.len() < 2_800,
            "compact 32x24 CBOR size {} >= 2.8 KB",
            cbor_compact.len()
        );
        let compact_rt = CropMetadata::from_cbor(&cbor_compact).unwrap();
        let crops_compact = compact_rt.compute_crops(1200, 900, &reqs, &[], &cfg);

        for (c_full, c_comp) in crops_full.iter().zip(crops_compact.iter()) {
            let a = c_full.expect("full crop");
            let b = c_comp.expect("compact crop");
            assert!((a.x as i32 - b.x as i32).abs() <= 24, "{a:?} vs {b:?}");
            assert!((a.y as i32 - b.y as i32).abs() <= 24, "{a:?} vs {b:?}");
            assert!((a.w as i32 - b.w as i32).abs() <= 24, "{a:?} vs {b:?}");
            assert!((a.h as i32 - b.h as i32).abs() <= 24, "{a:?} vs {b:?}");
        }

        // Also verify resolution independence: deriving crops for a 2400x1800 scaled image
        // scales the crop coordinates by 2x.
        let crops_2x = meta.compute_crops(2400, 1800, &reqs, &[], &cfg);
        for (c1, c2) in crops_full.iter().zip(crops_2x.iter()) {
            let a = c1.unwrap();
            let b = c2.unwrap();
            assert!((b.x as i32 - (a.x as i32 * 2)).abs() <= 2);
            assert!((b.y as i32 - (a.y as i32 * 2)).abs() <= 2);
            assert!((b.w as i32 - (a.w as i32 * 2)).abs() <= 2);
            assert!((b.h as i32 - (a.h as i32 * 2)).abs() <= 2);
        }
    }
}
