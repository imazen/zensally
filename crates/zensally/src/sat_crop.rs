//! Dual summed-area tables (`IntegralMap`) and the O(1)-per-candidate
//! arbitrary-aspect min/max crop solver from the zensally v0.2 design.
//!
//! `DualIntegralMap` fuses the Stride-8 multi-task tensor into two `u32`
//! fixed-point SATs plus the horizon profile:
//!
//! * `energy_sat` — Visual Saliency (Ch 0) + Foreground Matte (Ch 1) +
//!   face/body/eye landmark weights + [`crate::stream_prep`] classical
//!   OKLab/Sobel energy.
//! * `cut_penalty_sat` — silhouette contour gradient `‖∇M‖` (times
//!   [`MATTE_GRADIENT_WEIGHT`]) plus explicit boundary-cut penalties
//!   (face core boxes, eye-to-chin and shoulder intervals) so candidate
//!   crop edges never slice through them.
//!
//! Exact O(1) queries: [`DualIntegralMap::rect_energy`] is 4 lookups;
//! [`DualIntegralMap::rect_edge_cut_penalty`] evaluates the four border
//! strips = 16 lookups → 20 array lookups per candidate rectangle.
//!
//! [`solve_crop_sat`] scans every valid position of a handful of
//! aspect-conformant sizes — a few hundred candidates per call, well under
//! the 80 µs search budget — and maps the winner back to exact source pixel
//! coordinates. [`find_distant_subject_tile`] implements the
//! Tile-on-Demand trigger: a compact (<4% of frame) body/matte peak where
//! Face didn't fire yields a 256×256 native-resolution tile for a second
//! pass.
//!
//! `no_std + alloc` compatible: `core` math only, no heap allocation inside
//! the candidate scan (`Vec`s allocated once at `build`).

// Phase-1 internals: the public surface is consumed by the v0.2 pipeline and
// the detached bench harness, not yet by the crate's public API.
#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;

use crate::preprocess::LetterboxInfo;
use crate::stream_prep::{floor32, round32};
use crate::{CropRect, FaceRect, FocusRegion, SaliencyMap};

/// Fixed-point multiplier applied to `f32` channel values before they are
/// accumulated into the `u32` SATs.
pub const SAT_SCALE: f32 = 1024.0;

/// Multiplier applied to the per-cell `‖∇M‖` contour gradient before it is
/// folded into `cut_penalty_sat` — silhouette boundaries get real weight
/// against arbitrary edge placement.
pub const MATTE_GRADIENT_WEIGHT: f32 = 4.0;

/// `f64::floor` for `no_std` (truncation + fixup).
#[inline]
fn floor64(x: f64) -> f64 {
    let t = x as i64 as f64;
    if t > x { t - 1.0 } else { t }
}

/// `f64::round` for `no_std` (half away from zero).
#[inline]
fn round64(x: f64) -> f64 {
    floor64(x.abs() + 0.5).copysign(x)
}

/// `f32::ceil` for `no_std`.
#[inline]
fn ceil32(x: f32) -> f32 {
    let f = floor32(x);
    if f < x { f + 1.0 } else { f }
}

// ---------------------------------------------------------------------------
// Coordinate mapping: target pixels → source pixels
// ---------------------------------------------------------------------------

/// Affine mapping on one axis: `src_px = target_px * scale + offset`.
#[derive(Debug, Clone, Copy)]
pub struct AxisMap {
    /// Source pixels per target pixel on this axis.
    pub scale: f64,
    /// Translation in source pixels (e.g. `-pad_left / ratio`).
    pub offset: f64,
}

/// Grid→source geometry for [`solve_crop_sat`] and
/// [`find_distant_subject_tile`]. The stride-8 grid lives in "target"
/// coordinates (`target_w × target_h`); each cell spans 8 target pixels.
#[derive(Debug, Clone, Copy)]
pub struct TargetToSrc {
    /// Horizontal axis map.
    pub x: AxisMap,
    /// Vertical axis map.
    pub y: AxisMap,
    /// Source image width in pixels.
    pub src_w: u32,
    /// Source image height in pixels.
    pub src_h: u32,
    /// Analysis target width (grid cells span 8 of these pixels).
    pub target_w: u32,
    /// Analysis target height.
    pub target_h: u32,
}

impl TargetToSrc {
    /// Stretch mapping: source ↔ target differ by independent per-axis scale.
    pub fn stretch(src_w: u32, src_h: u32, target_w: u32, target_h: u32) -> Self {
        Self {
            x: AxisMap {
                scale: src_w as f64 / target_w as f64,
                offset: 0.0,
            },
            y: AxisMap {
                scale: src_h as f64 / target_h as f64,
                offset: 0.0,
            },
            src_w,
            src_h,
            target_w,
            target_h,
        }
    }

    /// Letterbox mapping from a [`LetterboxInfo`]: `src = (target − pad) /
    /// ratio` on each axis.
    pub fn letterbox(
        src_w: u32,
        src_h: u32,
        target_w: u32,
        target_h: u32,
        info: LetterboxInfo,
    ) -> Self {
        let inv = 1.0 / info.ratio as f64;
        Self {
            x: AxisMap {
                scale: inv,
                offset: -info.pad_left as f64 * inv,
            },
            y: AxisMap {
                scale: inv,
                offset: -info.pad_top as f64 * inv,
            },
            src_w,
            src_h,
            target_w,
            target_h,
        }
    }

    /// Pick [`stretch`](Self::stretch) or [`letterbox`](Self::letterbox)
    /// matching what [`crate::stream_prep::preprocess_nhwc_strided`] did.
    pub fn for_mode(
        src_w: u32,
        src_h: u32,
        target_w: u32,
        target_h: u32,
        mode: crate::preprocess::ResizeMode,
        info: LetterboxInfo,
    ) -> Self {
        match mode {
            crate::preprocess::ResizeMode::Stretch => {
                Self::stretch(src_w, src_h, target_w, target_h)
            }
            crate::preprocess::ResizeMode::Letterbox => {
                Self::letterbox(src_w, src_h, target_w, target_h, info)
            }
        }
    }

    /// Map a target-pixel coordinate to source pixels.
    #[inline]
    fn map_x(&self, v: f64) -> f64 {
        v * self.x.scale + self.x.offset
    }

    #[inline]
    fn map_y(&self, v: f64) -> f64 {
        v * self.y.scale + self.y.offset
    }

    /// Inverse of [`map_x`](Self::map_x) in *grid-cell* units:
    /// `cell = (src_px − offset) / scale / 8`.
    #[inline]
    fn src_to_cell_x(&self, src_px: f64) -> f64 {
        (src_px - self.x.offset) / self.x.scale / 8.0
    }

    /// Inverse of [`map_y`](Self::map_y) in grid-cell units.
    #[inline]
    fn src_to_cell_y(&self, src_px: f64) -> f64 {
        (src_px - self.y.offset) / self.y.scale / 8.0
    }
}

// ---------------------------------------------------------------------------
// DualIntegralMap
// ---------------------------------------------------------------------------

/// Optional input channels for [`DualIntegralMap::build`]. Every provided
/// slice must be exactly `grid_w * grid_h` `f32`s in row-major cell order;
/// `horizon_profile` must be `grid_h`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegralChannels<'a> {
    /// Ch 0: visual saliency heatmap.
    pub saliency: Option<&'a [f32]>,
    /// Ch 1: foreground subject matte.
    pub matte: Option<&'a [f32]>,
    /// Detected face/body/eye landmark weights.
    pub landmark_weights: Option<&'a [f32]>,
    /// Classical OKLab/Sobel structural energy from `stream_prep`.
    pub classical_energy: Option<&'a [f32]>,
    /// Explicit boundary-cut penalties (face core boxes, eye-to-chin bands).
    pub cut_penalty: Option<&'a [f32]>,
    /// Per-row horizontal edge continuity from `stream_prep`.
    pub horizon_profile: Option<&'a [f32]>,
}

/// Detected face with optional 5-point landmarks in percentage coordinates
/// (0.0–100.0 of the source image).
#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarkInput {
    /// Face bounding box in percentage coordinates (0.0–100.0).
    pub rect: FaceRect,
    /// Optional 5-point landmarks `[[x, y]; 5]` in percentage coordinates:
    /// `[right_eye, left_eye, nose_tip, right_mouth, left_mouth]`.
    pub landmarks: Option<[[f32; 2]; 5]>,
}

/// High-level crop input cues to fuse into a [`DualIntegralMap`].
///
/// This is the rich alternative to [`IntegralChannels`]: neural maps are
/// resampled from their native resolutions onto the stride-8 grid with
/// proper area averaging, faces carry optional 5-point landmarks for
/// eye-to-chin protection, and manual focus regions rasterize as
/// high-weight focal energy.
#[derive(Clone, Copy, Default)]
pub struct CropInputSources<'a> {
    /// Visual saliency heatmap (e.g. 128×128 from MicroSalNet v3ds w24).
    pub saliency: Option<&'a SaliencyMap>,
    /// Optional person/foreground segmentation matte (e.g. 256×256 from
    /// SelfieSeg). When absent but `saliency` is present, a pseudo-matte is
    /// derived from the squared saliency so silhouette-contour cut
    /// penalties still apply.
    pub matte: Option<&'a SaliencyMap>,
    /// Detected faces with optional 5-point landmarks.
    pub faces: &'a [FaceLandmarkInput],
    /// User-supplied manual focus regions (percentage coordinates).
    pub manual_focus: &'a [FocusRegion],
    /// Stride-8 classical OKLab chroma dispersion + Sobel energy from
    /// `stream_prep` (`grid_w * grid_h` cells).
    pub classical_energy: Option<&'a [f32]>,
    /// Per-row horizontal edge continuity from `stream_prep` (`grid_h`
    /// entries).
    pub horizon_profile: Option<&'a [f32]>,
}

/// A face or protected-focus box rasterized into grid-cell coordinates.
///
/// Fractional cell edges preserve sub-cell placement from the detector;
/// `headroom_y1`/`chin_y2` extend the face box upward/downward so solvers
/// can require forehead clearance and neck margin without re-deriving the
/// anatomy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceCellBox {
    /// Face box left edge (cells).
    pub fx1: f32,
    /// Face box top edge (cells).
    pub fy1: f32,
    /// Face box right edge (cells).
    pub fx2: f32,
    /// Face box bottom edge (cells).
    pub fy2: f32,
    /// Forehead headroom top: `fy1 − 0.35 * face_h` for faces, `fy1` for
    /// manual focus boxes.
    pub headroom_y1: f32,
    /// Chin/neck margin bottom: `fy2 + 0.25 * face_h` for faces, `fy2` for
    /// manual focus boxes.
    pub chin_y2: f32,
    /// Protected core left edge (cells): landmark x-span padded by 30%
    /// of the eye span for faces with landmarks (outer eye corners),
    /// otherwise the inner 76% of the box.
    pub px1: f32,
    /// Protected core top edge (cells): eye line − 18% of face height for
    /// landmarked faces (eyebrow clearance), else `fy1 + 0.15 * face_h`.
    pub py1: f32,
    /// Protected core right edge (cells).
    pub px2: f32,
    /// Protected core bottom edge (cells): mouth line + 8% of face height
    /// for landmarked faces (lower lip), else `fy2 + 0.15 * face_h`.
    pub py2: f32,
    /// Raw feature band left edge (cells): the landmark bbox itself for
    /// landmarked faces, else the core's own extent. When the padded
    /// core cannot fit a crop axis, the band is the inviolable region
    /// the crop must still contain on that axis.
    pub bx1: f32,
    /// Raw feature band top edge (cells).
    pub by1: f32,
    /// Raw feature band right edge (cells).
    pub bx2: f32,
    /// Raw feature band bottom edge (cells).
    pub by2: f32,
    /// Detector confidence (1.0 for manual focus regions).
    pub confidence: f32,
}

/// Dual summed-area tables over the stride-8 grid.
///
/// Both SATs are `(grid_w + 1) * (grid_h + 1)` `u32`s with row 0 / column 0
/// zeroed. Cell values are `f32` inputs clamped to `[0, cell_cap]` and
/// scaled by [`SAT_SCALE`]; `cell_cap` is derived so the table total can
/// never exceed `u32::MAX` (≈ 874 units/cell at 80×60).
pub struct DualIntegralMap {
    gw: usize,
    gh: usize,
    energy_sat: Vec<u32>,
    cut_sat: Vec<u32>,
    horizon: Vec<f32>,
    energy_total: u32,
    /// Subject bounding box in cells (x1, y1, x2, y2 exclusive), if any.
    /// `pub` for evaluation tooling; unreachable outside `zensally` since
    /// the module is `pub(crate)`.
    pub subject: Option<(usize, usize, usize, usize)>,
    subject_energy: u32,
    /// Detected faces in grid-cell coords (for solver face protection).
    pub faces: Vec<FaceCellBox>,
    /// Manual focus regions in grid-cell coords (solver containment terms).
    pub focus_boxes: Vec<FaceCellBox>,
}

impl DualIntegralMap {
    /// Build the dual SATs from the Stride-8 channels.
    ///
    /// `subject_threshold` selects the cells forming the subject bounding
    /// box: when `matte` is provided it is an absolute matte probability
    /// (e.g. 0.25); otherwise it is the fraction of the fused energy maximum
    /// used as the subject cutoff.
    pub fn build(ch: &IntegralChannels<'_>, gw: usize, gh: usize, subject_threshold: f32) -> Self {
        let n = gw * gh;
        let assert_len = |s: Option<&[f32]>, name: &str| {
            if let Some(s) = s {
                assert_eq!(s.len(), n, "{name} len {} != grid {gw}x{gh}", s.len());
            }
        };
        assert_len(ch.saliency, "saliency");
        assert_len(ch.matte, "matte");
        assert_len(ch.landmark_weights, "landmark_weights");
        assert_len(ch.classical_energy, "classical_energy");
        assert_len(ch.cut_penalty, "cut_penalty");
        if let Some(h) = ch.horizon_profile {
            assert_eq!(h.len(), gh, "horizon_profile len != {gh}");
        }

        // Per-cell fixed-point cap so n * cell <= u32::MAX, guaranteed.
        let cell_cap_fixed = if n == 0 {
            u32::MAX
        } else {
            u32::MAX / n as u32
        };
        let cell_cap = cell_cap_fixed as f32 / SAT_SCALE;
        let to_fixed = |v: f32| -> u32 {
            match v.partial_cmp(&0.0) {
                Some(core::cmp::Ordering::Greater) => round32(v.min(cell_cap) * SAT_SCALE) as u32,
                _ => 0, // zero, negative and NaN all quantize to zero
            }
        };

        let mut energy_f = alloc::vec![0.0f32; n];
        let mut cells_e = alloc::vec![0u32; n];
        let mut cells_c = alloc::vec![0u32; n];
        let mut max_e = 0.0f32;
        for i in 0..n {
            let e = ch.saliency.map_or(0.0, |s| s[i])
                + ch.matte.map_or(0.0, |s| s[i])
                + ch.landmark_weights.map_or(0.0, |s| s[i])
                + ch.classical_energy.map_or(0.0, |s| s[i]);
            energy_f[i] = e;
            max_e = max_e.max(e);
            cells_e[i] = to_fixed(e);

            let grad = ch.matte.map_or(0.0, |m| {
                let (x, y) = (i % gw, i / gw);
                let xl = if x == 0 { m[i] } else { m[i - 1] };
                let xr = if x + 1 == gw { m[i] } else { m[i + 1] };
                let yt = if y == 0 { m[i] } else { m[i - gw] };
                let yb = if y + 1 == gh { m[i] } else { m[i + gw] };
                0.5 * ((xr - xl).abs() + (yb - yt).abs())
            });
            cells_c[i] =
                to_fixed(ch.cut_penalty.map_or(0.0, |s| s[i]) + MATTE_GRADIENT_WEIGHT * grad);
        }

        // Subject bbox: matte cells over threshold, else fused-energy cells
        // over `subject_threshold * max`.
        let (sx1, sy1, sx2, sy2) = if let Some(m) = ch.matte {
            bbox_above(gw, gh, |i| m[i] >= subject_threshold)
        } else {
            let t = subject_threshold * max_e;
            bbox_above(gw, gh, |i| energy_f[i] >= t)
        };
        let subject = (sx2 > sx1 && sy2 > sy1).then_some((sx1, sy1, sx2, sy2));

        Self::assemble(
            gw,
            gh,
            cells_e,
            cells_c,
            ch.horizon_profile,
            subject,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Build a `DualIntegralMap` from rich crop cues: neural saliency and
    /// segmentation maps are area-resampled onto the grid, saliency is
    /// squared to suppress background haze, classical energy is
    /// percentile-normalized to a 0–0.15 tie-breaker, faces + 5-point
    /// landmarks rasterize focal energy and eye-to-chin cut penalties, and
    /// the subject bbox is derived from neighborhood-smoothed semantic
    /// foreground anchored on detected faces / manual focus.
    ///
    /// `gw`,`gh` are the stride-8 grid dims matching `to_src.target_*`/8;
    /// `subject_threshold` is the fraction of the semantic maximum that
    /// counts as foreground.
    pub fn from_crop_inputs(
        sources: &CropInputSources<'_>,
        gw: usize,
        gh: usize,
        to_src: &TargetToSrc,
        subject_threshold: f32,
    ) -> Self {
        let n = gw * gh;
        if let Some(e) = sources.classical_energy {
            assert!(e.len() >= n, "classical_energy len < grid {gw}x{gh}");
        }
        if let Some(h) = sources.horizon_profile {
            assert_eq!(h.len(), gh, "horizon_profile len != {gh}");
        }
        let (src_w, src_h) = (to_src.src_w as f64, to_src.src_h as f64);

        // Cell → source-pixel rect; a cell is "active" when it intersects
        // the source image (Letterbox padding is inactive → 0).
        let cell_active = |cx: usize, cy: usize| -> bool {
            let sx1 = to_src.map_x(cx as f64 * 8.0);
            let sx2 = to_src.map_x((cx + 1) as f64 * 8.0);
            let sy1 = to_src.map_y(cy as f64 * 8.0);
            let sy2 = to_src.map_y((cy + 1) as f64 * 8.0);
            sx2 > 0.0 && sy2 > 0.0 && sx1 < src_w && sy1 < src_h
        };

        // --- Squared saliency + matte, area-averaged onto the grid. ---
        // Saliency is normalized by the robust *peak* — the mean of the
        // top 0.5% of active cells (at least 8, floored at 0.15) — before
        // squaring. A compact subject under ~5% of the grid sits above a
        // p95 cutoff, which would saturate background haze to 1.0 and
        // stretch the subject bbox over the whole scene; the top-cell
        // mean tracks the true subject peak instead.
        let mut sal_sq = alloc::vec![0.0f32; n];
        if let Some(sal) = sources.saliency {
            resample_area_avg(
                &sal.data,
                sal.width,
                sal.height,
                to_src,
                gw,
                gh,
                &mut sal_sq,
            );
            let mut vals: Vec<f32> = (0..n)
                .filter(|&i| cell_active(i % gw, i / gw) && sal_sq[i] > 0.0)
                .map(|i| sal_sq[i])
                .collect();
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
                    *v = v.clamp(0.0, 1.0) * v.clamp(0.0, 1.0);
                }
            }
        }
        let mut matte = alloc::vec![0.0f32; n];
        if let Some(m) = sources.matte {
            resample_area_avg(&m.data, m.width, m.height, to_src, gw, gh, &mut matte);
            for v in &mut matte {
                *v = v.clamp(0.0, 1.0);
            }
        }

        // --- Classical energy: p95-normalized tie-breaker (0..=0.15). ---
        let mut classical = alloc::vec![0.0f32; n];
        if let Some(e) = sources.classical_energy {
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
                            classical[i] = (e[i].max(0.0) / p95).min(1.0) * 0.15;
                        }
                    }
                }
            }
        }

        // --- Faces: focal energy + landmark bands + cut penalties. ---
        let mut landmark_w = alloc::vec![0.0f32; n];
        let mut cut = alloc::vec![0.0f32; n];
        let mut faces: Vec<FaceCellBox> = Vec::with_capacity(sources.faces.len());
        for f in sources.faces {
            let pct_to_cell_x = |p: f32| {
                (to_src.src_to_cell_x(p as f64 / 100.0 * src_w) as f32).clamp(0.0, gw as f32)
            };
            let pct_to_cell_y = |p: f32| {
                (to_src.src_to_cell_y(p as f64 / 100.0 * src_h) as f32).clamp(0.0, gh as f32)
            };
            let (fx1, fy1) = (pct_to_cell_x(f.rect.x1), pct_to_cell_y(f.rect.y1));
            let (fx2, fy2) = (pct_to_cell_x(f.rect.x2), pct_to_cell_y(f.rect.y2));
            let (fw, fh) = (fx2 - fx1, fy2 - fy1);
            if fw <= 0.0 || fh <= 0.0 {
                continue;
            }
            let conf = f.rect.confidence.clamp(0.0, 1.0);
            let headroom_y1 = fy1 - 0.35 * fh;
            let chin_y2 = fy2 + 0.25 * fh;

            // Face core + eye-to-mouth landmark core.
            add_rect(&mut landmark_w, gw, gh, fx1, fy1, fx2, fy2, 6.0 * conf);
            let (band_y1, band_y2, px1, px2, raw_band);
            if let Some(lm) = f.landmarks {
                let mut lx1 = f32::MAX;
                let mut ly1 = f32::MAX;
                let mut lx2 = f32::MIN;
                let mut ly2 = f32::MIN;
                for p in &lm {
                    lx1 = lx1.min(pct_to_cell_x(p[0]));
                    ly1 = ly1.min(pct_to_cell_y(p[1]));
                    lx2 = lx2.max(pct_to_cell_x(p[0]));
                    ly2 = ly2.max(pct_to_cell_y(p[1]));
                }
                add_rect(&mut landmark_w, gw, gh, lx1, ly1, lx2, ly2, 10.0 * conf);
                let eye_y = pct_to_cell_y(lm[0][1]).min(pct_to_cell_y(lm[1][1]));
                let mouth_y = pct_to_cell_y(lm[3][1]).max(pct_to_cell_y(lm[4][1]));
                // Landmark points are feature *centers* (pupils, mouth
                // corners): pad to the anatomical core — outer eye
                // corners + eyebrow clearance on top, lower lip on the
                // bottom — so a crop edge can't satisfy containment
                // while still slicing an eye corner or the brow line.
                let eye_span = (lx2 - lx1).max(0.30 * fw);
                px1 = (lx1 - 0.30 * eye_span)
                    .max(fx1 + 0.06 * fw)
                    .min(fx2 - 0.06 * fw);
                px2 = (lx2 + 0.30 * eye_span).min(fx2 - 0.06 * fw);
                band_y1 = (eye_y - 0.18 * fh).max(fy1 + 0.04 * fh);
                band_y2 = (mouth_y + 0.08 * fh).min(fy2);
                raw_band = (
                    lx1.clamp(0.0, gw as f32),
                    ly1.clamp(0.0, gh as f32),
                    lx2.clamp(0.0, gw as f32),
                    ly2.clamp(0.0, gh as f32),
                );
            } else {
                band_y1 = fy1 + 0.15 * fh;
                band_y2 = fy2 + 0.15 * fh;
                px1 = fx1 + 0.12 * fw;
                px2 = fx2 - 0.12 * fw;
                raw_band = (px1, band_y1, px2, band_y2);
            }
            faces.push(FaceCellBox {
                fx1,
                fy1,
                fx2,
                fy2,
                headroom_y1,
                chin_y2,
                px1,
                py1: band_y1.clamp(0.0, gh as f32),
                px2,
                py2: band_y2.clamp(0.0, gh as f32),
                bx1: raw_band.0,
                by1: raw_band.1,
                bx2: raw_band.2,
                by2: raw_band.3,
                confidence: conf,
            });
            // Upper-torso support region below the chin.
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
            // Cut penalties: a light penalty on the outer head/hair/cheek
            // box and a heavy one concentrated on the anatomical
            // eye→mouth core — a crop tighter than the head then prefers
            // resting its edges in cheek/forehead margin rather than
            // slicing features (a uniform-width penalty rewarded shifting
            // the far edge out of the box instead).
            add_rect(&mut cut, gw, gh, fx1, headroom_y1, fx2, chin_y2, 8.0 * conf);
            add_rect(&mut cut, gw, gh, px1, band_y1, px2, band_y2, 45.0 * conf);
        }

        // Manual focus regions: high focal energy + protected boxes.
        let mut focus_boxes: Vec<FaceCellBox> = Vec::with_capacity(sources.manual_focus.len());
        for r in sources.manual_focus {
            let fx1 =
                (to_src.src_to_cell_x(r.x1 as f64 / 100.0 * src_w) as f32).clamp(0.0, gw as f32);
            let fy1 =
                (to_src.src_to_cell_y(r.y1 as f64 / 100.0 * src_h) as f32).clamp(0.0, gh as f32);
            let fx2 =
                (to_src.src_to_cell_x(r.x2 as f64 / 100.0 * src_w) as f32).clamp(0.0, gw as f32);
            let fy2 =
                (to_src.src_to_cell_y(r.y2 as f64 / 100.0 * src_h) as f32).clamp(0.0, gh as f32);
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

        // --- Unified foreground mask `fg`. ---
        // `sal_matte` is the squared-saliency pseudo-matte; when a
        // segmentation matte is also present it is gated by local
        // saliency support so a person-only segmenter (SelfieSeg) firing
        // on peripheral limbs or bystanders can't hijack non-person
        // photos (a waiter's hand in a food shot). Cells inside a
        // detected face (+ upper torso) or manual focus box get full
        // support so detected people keep their matte where saliency is
        // weak. Matte without saliency is trusted as-is.
        let mut fg = alloc::vec![0.0f32; n];
        for (g, &s) in fg.iter_mut().zip(sal_sq.iter()) {
            *g = ((s - 0.12) / 0.60).clamp(0.0, 1.0);
        }
        if sources.matte.is_some() {
            if sources.saliency.is_some() {
                let mut support = alloc::vec![0.0f32; n];
                for (v, &s) in support.iter_mut().zip(sal_sq.iter()) {
                    *v = s.sqrt();
                }
                for f in &faces {
                    let (fw, fh) = (f.fx2 - f.fx1, f.fy2 - f.fy1);
                    raise_rect(
                        &mut support,
                        gw,
                        gh,
                        f.fx1 - 0.35 * fw,
                        f.headroom_y1,
                        f.fx2 + 0.35 * fw,
                        f.fy2 + 1.25 * fh,
                        1.0,
                    );
                }
                for f in &focus_boxes {
                    raise_rect(&mut support, gw, gh, f.fx1, f.fy1, f.fx2, f.fy2, 1.0);
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

        // --- Fuse channels (energy) and quantize like `build`. ---
        let cell_cap_fixed = if n == 0 {
            u32::MAX
        } else {
            u32::MAX / n as u32
        };
        let cell_cap = cell_cap_fixed as f32 / SAT_SCALE;
        let to_fixed = |v: f32| -> u32 {
            match v.partial_cmp(&0.0) {
                Some(core::cmp::Ordering::Greater) => round32(v.min(cell_cap) * SAT_SCALE) as u32,
                _ => 0,
            }
        };

        let mut cells_e = alloc::vec![0u32; n];
        let mut cells_c = alloc::vec![0u32; n];
        for i in 0..n {
            let e = sal_sq[i] + fg[i] + landmark_w[i] + classical[i];
            cells_e[i] = to_fixed(e);
            let (x, y) = (i % gw, i / gw);
            let xl = if x == 0 { fg[i] } else { fg[i - 1] };
            let xr = if x + 1 == gw { fg[i] } else { fg[i + 1] };
            let yt = if y == 0 { fg[i] } else { fg[i - gw] };
            let yb = if y + 1 == gh { fg[i] } else { fg[i + gw] };
            let grad = 0.5 * ((xr - xl).abs() + (yb - yt).abs());
            // Interior-bisection penalty: ‖∇fg‖ ≈ 0 inside a uniform
            // salient object, so a border strip through its hot core
            // would otherwise pay nothing. Hot cells make crop edges
            // prefer background margins over slicing a salient object.
            let hot = 5.0 * ((sal_sq[i] - 0.35) / 0.65).clamp(0.0, 1.0);
            cells_c[i] = to_fixed(cut[i] + hot + MATTE_GRADIENT_WEIGHT * grad);
        }

        // --- Outlier-resistant subject bbox. ---
        // Semantic foreground = max(sal², matte); a cell only counts when
        // its own value clears the threshold AND its 3×3 mean clears 65%
        // of it, killing isolated speckles.
        let mut sem = alloc::vec![0.0f32; n];
        let mut max_sem = 0.0f32;
        for i in 0..n {
            if cell_active(i % gw, i / gw) {
                let s = sal_sq[i].max(fg[i]);
                sem[i] = s;
                max_sem = max_sem.max(s);
            }
        }
        // Mean over the full 3×3 window — out-of-bounds cells count as 0,
        // so a lone hot cell on the grid edge can't fake a neighborhood.
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
            // Anchored bbox: faces (headroom + upper torso) ∪ focus ∪
            // high-confidence smoothed semantic cells.
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
            let t = subject_threshold * max_sem;
            let (sx1, sy1, sx2, sy2) = bbox_above(gw, gh, |i| smoothed(i, t));
            (sx2 > sx1 && sy2 > sy1).then_some((sx1, sy1, sx2, sy2))
        };

        Self::assemble(
            gw,
            gh,
            cells_e,
            cells_c,
            sources.horizon_profile,
            subject,
            faces,
            focus_boxes,
        )
    }

    /// Shared tail of `build`/`from_crop_inputs`: accumulate the per-cell
    /// fixed-point columns into the two SATs and finish the struct.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        gw: usize,
        gh: usize,
        cells_e: Vec<u32>,
        cells_c: Vec<u32>,
        horizon_profile: Option<&[f32]>,
        subject: Option<(usize, usize, usize, usize)>,
        faces: Vec<FaceCellBox>,
        focus_boxes: Vec<FaceCellBox>,
    ) -> Self {
        let w1 = gw + 1;
        let mut energy_sat = alloc::vec![0u32; w1 * (gh + 1)];
        let mut cut_sat = alloc::vec![0u32; w1 * (gh + 1)];
        for y in 0..gh {
            let mut row_e = 0u64;
            let mut row_c = 0u64;
            for x in 0..gw {
                row_e += cells_e[y * gw + x] as u64;
                row_c += cells_c[y * gw + x] as u64;
                energy_sat[(y + 1) * w1 + x + 1] =
                    (energy_sat[y * w1 + x + 1] as u64 + row_e) as u32;
                cut_sat[(y + 1) * w1 + x + 1] = (cut_sat[y * w1 + x + 1] as u64 + row_c) as u32;
            }
        }
        let energy_total = cells_e
            .iter()
            .map(|&v| v as u64)
            .sum::<u64>()
            .min(u32::MAX as u64) as u32;
        let subject_energy = subject.map_or(0, |(x1, y1, x2, y2)| {
            sat_rect(&energy_sat, w1, x1, y1, x2, y2)
        });
        Self {
            gw,
            gh,
            energy_sat,
            cut_sat,
            horizon: horizon_profile.map_or_else(|| alloc::vec![0.0; gh], |h| h.to_vec()),
            energy_total,
            subject,
            subject_energy,
            faces,
            focus_boxes,
        }
    }

    /// Grid width in cells.
    pub fn grid_w(&self) -> usize {
        self.gw
    }
    /// Grid height in cells.
    pub fn grid_h(&self) -> usize {
        self.gh
    }
    /// Total fused energy (fixed-point units).
    pub fn energy_total(&self) -> u32 {
        self.energy_total
    }
    /// Per-row horizon continuity profile (`grid_h` entries).
    pub fn horizon_profile(&self) -> &[f32] {
        &self.horizon
    }
    /// Subject bounding box in cells, (x1, y1, x2, y2) exclusive.
    pub fn subject_bbox(&self) -> Option<(usize, usize, usize, usize)> {
        self.subject
    }
    /// Detected faces rasterized to grid-cell coords (empty unless built
    /// via [`from_crop_inputs`](Self::from_crop_inputs)).
    pub fn faces(&self) -> &[FaceCellBox] {
        &self.faces
    }
    /// Manual focus regions rasterized to grid-cell coords.
    pub fn focus_boxes(&self) -> &[FaceCellBox] {
        &self.focus_boxes
    }

    /// Energy inside cell rectangle `[x1..x2) × [y1..y2)` — 4 lookups.
    /// Coordinates are clamped to the grid; empty rects return 0.
    #[inline]
    pub fn rect_energy(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u32 {
        let (x1, x2) = (x1.min(self.gw), x2.min(self.gw));
        let (y1, y2) = (y1.min(self.gh), y2.min(self.gh));
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        sat_rect(&self.energy_sat, self.gw + 1, x1, y1, x2, y2)
    }

    /// Cut penalty inside an arbitrary rect (used for subject intersection).
    #[inline]
    pub fn rect_cut(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> u32 {
        let (x1, x2) = (x1.min(self.gw), x2.min(self.gw));
        let (y1, y2) = (y1.min(self.gh), y2.min(self.gh));
        if x2 <= x1 || y2 <= y1 {
            return 0;
        }
        sat_rect(&self.cut_sat, self.gw + 1, x1, y1, x2, y2)
    }

    /// Cut penalty under the four border strips of `[x1..x2) × [y1..y2)`:
    /// full-width `band`-cell strips on top and bottom, plus the remaining
    /// left/right strips between them — 4 SAT queries = 16 lookups.
    #[inline]
    pub fn rect_edge_cut_penalty(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        band: usize,
    ) -> u32 {
        let (x1, x2) = (x1.min(self.gw), x2.min(self.gw));
        let (y1, y2) = (y1.min(self.gh), y2.min(self.gh));
        if x2 <= x1 || y2 <= y1 || band == 0 {
            return 0;
        }
        let s = &self.cut_sat;
        let w1 = self.gw + 1;
        let yt = (y1 + band).min(y2);
        let yb = y2.saturating_sub(band).max(yt);
        let xl = (x1 + band).min(x2);
        let xr = x2.saturating_sub(band).max(xl);
        sat_rect(s, w1, x1, y1, x2, yt)
            + sat_rect(s, w1, x1, yb, x2, y2)
            + sat_rect(s, w1, x1, yt, xl, yb)
            + sat_rect(s, w1, xr, yt, x2, yb)
    }
}

/// Area-average a `mw × mh` heatmap (e.g. a 128×128 saliency map or
/// 256×256 matte) into the stride-8 grid: each grid cell accumulates the
/// overlap-weighted mean of the map pixels under its source-pixel rect.
/// Cells wholly outside the source (letterbox padding) stay 0.
fn resample_area_avg(
    data: &[f32],
    mw: u32,
    mh: u32,
    to_src: &TargetToSrc,
    gw: usize,
    gh: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), gw * gh, "resample out len != grid");
    assert_eq!(
        data.len(),
        mw as usize * mh as usize,
        "resample map len != {mw}x{mh}"
    );
    let (mw, mh) = (mw as usize, mh as usize);
    if mw == 0 || mh == 0 {
        return;
    }
    let kx = mw as f64 / to_src.src_w as f64;
    let ky = mh as f64 / to_src.src_h as f64;
    let (src_w, src_h) = (to_src.src_w as f64, to_src.src_h as f64);
    for cy in 0..gh {
        for cx in 0..gw {
            let sx1 = to_src.map_x(cx as f64 * 8.0).clamp(0.0, src_w);
            let sx2 = to_src.map_x((cx + 1) as f64 * 8.0).clamp(0.0, src_w);
            let sy1 = to_src.map_y(cy as f64 * 8.0).clamp(0.0, src_h);
            let sy2 = to_src.map_y((cy + 1) as f64 * 8.0).clamp(0.0, src_h);
            if sx2 <= sx1 || sy2 <= sy1 {
                continue;
            }
            // Map-pixel-space rect (fractional).
            let (rx1, rx2) = (sx1 * kx, sx2 * kx);
            let (ry1, ry2) = (sy1 * ky, sy2 * ky);
            let (mut sum, mut wsum) = (0.0f64, 0.0f64);
            let py1 = floor64(ry1) as usize;
            let py2 = (floor64(ry2) as usize + 1).min(mh);
            for py in py1..py2 {
                let wy = (ry2.min(py as f64 + 1.0) - ry1.max(py as f64)).max(0.0);
                if wy <= 0.0 {
                    continue;
                }
                let px1 = floor64(rx1) as usize;
                let px2 = (floor64(rx2) as usize + 1).min(mw);
                for px in px1..px2 {
                    let wx = (rx2.min(px as f64 + 1.0) - rx1.max(px as f64)).max(0.0);
                    if wx > 0.0 {
                        sum += wx * wy * data[py * mw + px] as f64;
                        wsum += wx * wy;
                    }
                }
            }
            if wsum > 0.0 {
                out[cy * gw + cx] = (sum / wsum) as f32;
            }
        }
    }
}

/// Rasterize a fractional cell rect into a `gw × gh` grid by adding `w` to
/// every integer cell the rect overlaps.
#[allow(clippy::too_many_arguments)]
fn add_rect(buf: &mut [f32], gw: usize, gh: usize, x1: f32, y1: f32, x2: f32, y2: f32, w: f32) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let ix1 = (floor32(x1) as i32).clamp(0, gw as i32) as usize;
    let iy1 = (floor32(y1) as i32).clamp(0, gh as i32) as usize;
    let ix2 = (ceil32(x2) as i32).clamp(0, gw as i32) as usize;
    let iy2 = (ceil32(y2) as i32).clamp(0, gh as i32) as usize;
    for y in iy1..iy2 {
        for x in ix1..ix2 {
            buf[y * gw + x] += w;
        }
    }
}

/// Raise every cell intersecting `(x1,y1)..(x2,y2)` to at least `v`.
#[allow(clippy::too_many_arguments)]
fn raise_rect(buf: &mut [f32], gw: usize, gh: usize, x1: f32, y1: f32, x2: f32, y2: f32, v: f32) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let ix1 = (floor32(x1) as i32).clamp(0, gw as i32) as usize;
    let iy1 = (floor32(y1) as i32).clamp(0, gh as i32) as usize;
    let ix2 = (ceil32(x2) as i32).clamp(0, gw as i32) as usize;
    let iy2 = (ceil32(y2) as i32).clamp(0, gh as i32) as usize;
    for y in iy1..iy2 {
        for x in ix1..ix2 {
            let i = y * gw + x;
            if v > buf[i] {
                buf[i] = v;
            }
        }
    }
}

/// `sat[y2*w + x2] - sat[y1*w + x2] - sat[y2*w + x1] + sat[y1*w + x1]` in
/// `u64` (differences are exact and non-negative by SAT construction).
#[inline]
fn sat_rect(sat: &[u32], w1: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> u32 {
    let a = sat[y2 * w1 + x2] as u64 + sat[y1 * w1 + x1] as u64;
    let b = sat[y1 * w1 + x2] as u64 + sat[y2 * w1 + x1] as u64;
    (a - b) as u32
}

/// Bounding box of cells where `pred` holds, exclusive coords.
fn bbox_above(gw: usize, gh: usize, pred: impl Fn(usize) -> bool) -> (usize, usize, usize, usize) {
    let (mut x1, mut y1, mut x2, mut y2) = (gw, gh, 0usize, 0usize);
    for y in 0..gh {
        for x in 0..gw {
            if pred(y * gw + x) {
                x1 = x1.min(x);
                y1 = y1.min(y);
                x2 = x2.max(x + 1);
                y2 = y2.max(y + 1);
            }
        }
    }
    (x1, y1, x2, y2)
}

// ---------------------------------------------------------------------------
// solve_crop_sat
// ---------------------------------------------------------------------------

/// Crop strategy, mirroring `zenlayout::smart_crop::CropMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatCropMode {
    /// Largest crop at the target ratio ("minimum content removal").
    /// Positions maximize captured energy minus boundary cut penalty, plus
    /// rule-of-thirds focal and horizon alignment bonuses; near-full scale
    /// steps down to [`MIN_SCALE_STEP`] are evaluated so the solver can
    /// dodge a severe silhouette/chin cut.
    Minimal,
    /// Tightest crop wrapping the primary subject / face region ("maximum
    /// zoom"), padded by `zoom_padding`, bounded below by
    /// `min_scale_floor`, penalizing uncaptured subject energy and border
    /// cuts.
    Maximal,
}

/// Largest `Minimal` scale reduction considered when dodging cut penalties.
pub const MIN_SCALE_STEP: f64 = 0.85;

/// Tuning knobs for [`solve_crop_sat`]. All score terms are normalized by
/// the map's total energy, so the defaults are unit-free.
#[derive(Debug, Clone)]
pub struct SatCropConfig {
    /// Border strip width in cells for `rect_edge_cut_penalty` (default 1).
    pub band: usize,
    /// Weight on border cut penalty (default 4.0 — a band of strong
    /// penalties dominates modest energy gains).
    pub cut_weight: f32,
    /// `Maximal`: weight on uncaptured subject energy (default 2.0).
    pub uncaptured_weight: f32,
    /// `Maximal`: weight on normalized crop area — keeps the winner tight
    /// (default 0.15).
    pub area_weight: f32,
    /// `Minimal`: bonus for energy near the crop's interior third lines
    /// (default 0.25).
    pub thirds_weight: f32,
    /// `Minimal`: bonus for placing horizon-rich rows on the crop's interior
    /// third lines (default 0.5).
    pub horizon_weight: f32,
    /// `Maximal`: padding around the subject as a fraction of subject size
    /// (default 0.5).
    pub zoom_padding: f32,
    /// `Maximal`: minimum crop size as a fraction of the largest fit
    /// (default 0.3, mirroring `zenlayout`'s 30% floor).
    pub min_scale_floor: f32,
}

impl Default for SatCropConfig {
    fn default() -> Self {
        Self {
            band: 1,
            cut_weight: 4.0,
            uncaptured_weight: 2.0,
            area_weight: 0.15,
            thirds_weight: 0.25,
            horizon_weight: 0.5,
            zoom_padding: 0.5,
            min_scale_floor: 0.3,
        }
    }
}

/// Solve for the best crop rectangle at `target_aspect` in `mode`.
///
/// `map` is the built [`DualIntegralMap`]; `to_src` converts grid
/// coordinates to source pixels (see [`TargetToSrc`]). Returns a
/// [`CropRect`] in source pixels with the exact requested aspect ratio
/// (integer-snapped, in bounds), or `None` for degenerate inputs.
pub fn solve_crop_sat(
    map: &DualIntegralMap,
    to_src: &TargetToSrc,
    target_aspect: (u32, u32),
    mode: SatCropMode,
    cfg: &SatCropConfig,
) -> Option<CropRect> {
    let (gw, gh) = (map.gw, map.gh);
    let (aw, ah) = target_aspect;
    if gw == 0 || gh == 0 || aw == 0 || ah == 0 {
        return None;
    }

    // cells_w / cells_h that yields aspect aw:ah in *source* pixels:
    // (wc*8*sx) / (hc*8*sy) = aw/ah  →  wc/hc = (aw*sy)/(ah*sx).
    let ca = (aw as f64 * to_src.y.scale) / (ah as f64 * to_src.x.scale);

    // Largest aspect-conformant rect in cells.
    let (wmax, hmax) = {
        let mut w = gw as f64;
        let mut h = floor64(w / ca);
        if h > gh as f64 {
            h = gh as f64;
            w = floor64(h * ca);
        }
        (w.max(1.0) as usize, h.max(1.0) as usize)
    };

    let subject = map
        .subject
        .unwrap_or((gw / 4, gh / 4, gw * 3 / 4, gh * 3 / 4));

    // Candidate sizes: deduplicated (w, h) cell dims at the aspect.
    let mut sizes: Vec<(usize, usize)> = Vec::new();
    {
        let mut push_size = |w: f64| {
            let w0 = (round64(w) as usize).clamp(1, gw);
            let h0 = (round64(w0 as f64 / ca) as usize).clamp(1, gh);
            // If h hit the grid clamp, re-fit w to stay aspect-conformant.
            let w = if h0 == gh && (h0 as f64 * ca) < w0 as f64 {
                (round64(h0 as f64 * ca) as usize).clamp(1, gw)
            } else {
                w0
            };
            if !sizes.contains(&(w, h0)) {
                sizes.push((w, h0));
            }
        };
        match mode {
            SatCropMode::Minimal => {
                // Near-full-scale candidates only: the +0.12 scale
                // retention bonus in `score` keeps the winner at 1.0
                // unless stepping down genuinely dodges a boundary cut.
                for s in [1.0f64, 0.95, 0.90, MIN_SCALE_STEP] {
                    push_size(wmax as f64 * s);
                }
            }
            SatCropMode::Maximal => {
                let (bx1, by1, bx2, by2) = subject;
                let pad_x = (bx2 - bx1) as f64 * cfg.zoom_padding as f64;
                let pad_y = (by2 - by1) as f64 * cfg.zoom_padding as f64;
                let pw = (bx2 - bx1) as f64 + 2.0 * pad_x;
                let ph = (by2 - by1) as f64 + 2.0 * pad_y;
                // Three fit seeds so narrow aspects still zoom: covering a
                // wide subject's *width* at 9:16 would force h > gh and
                // collapse to a single full-frame candidate; fitting its
                // *height* (w_fit_h) or width (w_fit_w) keeps real zoom
                // levels in the set.
                let mut seeds = alloc::vec![
                    pw.max(ph * ca), // covers the whole padded subject
                    ph * ca,         // fits padded subject height
                    pw,              // fits padded subject width
                ];
                // Portrait framing seed from the highest-confidence face:
                // face height × 2.8 gives eyes-up headroom + torso.
                if let Some(f) = map.faces.iter().max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(core::cmp::Ordering::Equal)
                }) {
                    seeds.push((f.fy2 - f.fy1) as f64 * 2.8 * ca);
                }
                // Zoom bounds: never below the min-scale floor, never at
                // the full-frame size unless that is the floor itself.
                // The 0.75 cap keeps `area(Maximal) <= 0.85 * area(Minimal)`
                // even when `Minimal` itself steps down to 0.85 to dodge a
                // boundary cut (0.75² < 0.85 × 0.85²).
                let fw = round64(wmax as f64 * cfg.min_scale_floor as f64).clamp(1.0, gw as f64);
                let max_zoom_w = round64(wmax as f64 * 0.75).max(fw);
                for &sd in &seeds {
                    for m in [0.75f64, 0.9, 1.0, 1.15, 1.35, 1.6] {
                        let w = round64(sd * m).clamp(fw, max_zoom_w);
                        let h = round64(w / ca).clamp(1.0, gh as f64);
                        let (w, h) = (w as usize, h as usize);
                        if !sizes.contains(&(w, h)) {
                            sizes.push((w, h));
                        }
                    }
                }
            }
        }
    }

    let norm = map.energy_total as f32 + 1.0;
    let horizon_sum: f32 = map.horizon.iter().sum::<f32>() + 1e-6;
    let band = cfg.band.max(1);
    let (bx1, by1, bx2, by2) = subject;

    // Score one candidate rect (all energy terms normalized by the total).
    let score = |x: usize, y: usize, w: usize, h: usize| -> f32 {
        let captured = map.rect_energy(x, y, x + w, y + h) as f32 / norm;
        let cut = map.rect_edge_cut_penalty(x, y, x + w, y + h, band) as f32 / norm;
        let mut s = captured - cfg.cut_weight * cut;
        if mode == SatCropMode::Minimal {
            // Scale retention: keep the crop at full scale unless a step
            // down genuinely buys cut-avoidance or captures a face.
            s += 0.12 * (w * h) as f32 / (wmax * hmax) as f32;
        }
        if cfg.thirds_weight > 0.0 && w >= 3 && h >= 3 {
            // Energy near the four interior third-line intersections.
            let mut focal = 0u32;
            for &tx in &[x + w / 3, x + 2 * w / 3] {
                for &ty in &[y + h / 3, y + 2 * h / 3] {
                    focal +=
                        map.rect_energy(tx.saturating_sub(1), ty.saturating_sub(1), tx + 1, ty + 1);
                }
            }
            s += cfg.thirds_weight * (focal as f32 / norm - 0.25).max(0.0);
        }
        if cfg.horizon_weight > 0.0 && !map.horizon.is_empty() && h >= 3 {
            let r1 = (y + h / 3).min(gh - 1);
            let r2 = (y + 2 * h / 3).min(gh - 1);
            s += cfg.horizon_weight * (map.horizon[r1] + map.horizon[r2]) / horizon_sum;
        }
        if mode == SatCropMode::Maximal {
            let ix1 = x.max(bx1);
            let iy1 = y.max(by1);
            let ix2 = (x + w).min(bx2);
            let iy2 = (y + h).min(by2);
            let captured_subj = map.rect_energy(ix1, iy1, ix2, iy2) as f32 / norm;
            let uncaptured = (map.subject_energy as f32 / norm - captured_subj).max(0.0);
            let area = (w * h) as f32 / (gw * gh) as f32;
            s += -cfg.uncaptured_weight * uncaptured - cfg.area_weight * area;
        }
        // Protected boxes (faces + manual focus). Face checks use the
        // *aspect-corrected* rect `grid_rect_to_src` will emit — centered
        // on (x,y,w,h) and shrunk to the exact target aspect — so a scored
        // "containment" can't be clipped away by the final aspect snap.
        // Terms weight confidence by √face-area so the dominant face
        // outranks background faces when not all fit.
        let ew = (w as f64).min(h as f64 * ca) as f32;
        let eh = (h as f64).min(w as f64 / ca) as f32;
        let ex1 = x as f32 + (w as f32 - ew) * 0.5;
        let ey1 = y as f32 + (h as f32 - eh) * 0.5;
        let ex2 = ex1 + ew;
        let ey2 = ey1 + eh;
        for f in map.faces.iter().chain(map.focus_boxes.iter()) {
            let weight = f.confidence * ((f.fx2 - f.fx1) * (f.fy2 - f.fy1)).sqrt();
            // Horizontal symmetry on the feature-core center — only for
            // crops *narrower than the head*: those center on the eyes
            // instead of drifting onto one cheek. A wide crop containing
            // several faces can't center on each; normalizing by `fw`
            // would make multi-face coverage prohibitively expensive.
            let fox = (ex2.min(f.fx2) - ex1.max(f.fx1)).max(0.0);
            let foy = (ey2.min(f.fy2) - ey1.max(f.fy1)).max(0.0);
            if fox > 0.0 && foy > 0.0 && ew < f.fx2 - f.fx1 {
                let pcx = 0.5 * (f.px1 + f.px2);
                s -= 1.5 * weight * ((0.5 * (ex1 + ex2) - pcx).abs() / (f.fx2 - f.fx1).max(1.0));
            }
            // Protected core (eye→mouth padded core / focus box):
            // coverage of the padded core is rewarded for every
            // intersecting candidate — full containment scores frac = 1,
            // strictly above every near-miss. The containment target is
            // per-axis: the padded core where it fits the effective
            // crop, else the raw feature band — so a wide crop that
            // can't keep brow..chin still keeps eyes..mouth, and a
            // narrow crop keeps the eye span. Slicing the raw band on
            // an axis where it fits is the hard veto; shaving only the
            // pad margin is the soft near-veto.
            let cox = (ex2.min(f.px2) - ex1.max(f.px1)).max(0.0);
            let coy = (ey2.min(f.py2) - ey1.max(f.py1)).max(0.0);
            let core_area = ((f.px2 - f.px1) * (f.py2 - f.py1)).max(1e-6);
            if cox > 0.0 && coy > 0.0 {
                s += 6.0 * weight * (cox * coy / core_area);
                let (tx1, tx2) = if (f.px2 - f.px1) <= ew {
                    (f.px1, f.px2)
                } else {
                    (f.bx1, f.bx2)
                };
                let (ty1, ty2) = if (f.py2 - f.py1) <= eh {
                    (f.py1, f.py2)
                } else {
                    (f.by1, f.by2)
                };
                let prot_in = ex1 <= tx1 && ey1 <= ty1 && ex2 >= tx2 && ey2 >= ty2;
                if !prot_in {
                    let band_cut = ((f.bx2 - f.bx1) <= ew && !(ex1 <= f.bx1 && ex2 >= f.bx2))
                        || ((f.by2 - f.by1) <= eh && !(ey1 <= f.by1 && ey2 >= f.by2));
                    s -= if band_cut { 25.0 } else { 15.0 } * weight;
                    continue;
                }
            }
            let box_in = ex1 <= f.fx1 && ey1 <= f.fy1 && ex2 >= f.fx2 && ey2 >= f.fy2;
            if box_in {
                s += 1.5 * weight;
                let fcy = 0.5 * (f.fy1 + f.fy2);
                if ey1 <= f.headroom_y1 && fcy >= ey1 + 0.20 * eh && fcy <= ey1 + 0.48 * eh {
                    s += 0.35 * weight;
                }
            } else if (f.fx2 - f.fx1) <= ew && (f.fy2 - f.fy1) <= eh {
                // Whole face would fit but the crop excludes it entirely.
                if fox <= 0.0 || foy <= 0.0 {
                    s -= 3.0 * weight;
                }
            }
        }
        // Subject apex protection: a crop shorter than a tall subject
        // must anchor at/above the subject's crown and trim the legs —
        // never decapitate. The penalty ramps over the top 35% of the
        // subject; a small bonus rewards keeping ≥2/3 of the body with
        // headroom. Uses the effective aspect-corrected rect.
        if let Some((_sx1, sy1, _sx2, sy2)) = map.subject {
            let bh = sy2 - sy1;
            if bh >= 4 && sy1 >= 1 && ey2 > sy1 as f32 && ey1 < sy2 as f32 {
                let subj_e = map.subject_energy as f32 / norm;
                if ey1 > sy1 as f32 {
                    let clipped = ((ey1 - sy1 as f32) / (bh as f32 * 0.35).max(1.0)).min(1.0);
                    s -= 0.45 * subj_e * clipped;
                } else if ey1 <= (sy1 - 1) as f32 && ey2 >= (sy1 + (bh * 2 / 3).max(1)) as f32 {
                    s += 0.08 * subj_e;
                }
            }
        }
        s
    };

    let mut best: Option<(f32, usize, usize, usize, usize)> = None;
    let pad_x = round64((bx2 - bx1) as f64 * cfg.zoom_padding as f64) as usize;
    let pad_y = round64((by2 - by1) as f64 * cfg.zoom_padding as f64) as usize;

    for &(w, h) in &sizes {
        if w > gw || h > gh {
            continue;
        }
        let (x_lo, x_hi, y_lo, y_hi) = match mode {
            SatCropMode::Minimal => (0usize, gw - w, 0usize, gh - h),
            SatCropMode::Maximal => {
                // Positions overlapping the padded subject box.
                let lo = (bx2 + pad_x).saturating_sub(w);
                let hi = bx1.saturating_sub(pad_x).min(gw - w);
                let (x_lo, x_hi) = if lo <= hi { (lo, hi) } else { (0, gw - w) };
                let lo = (by2 + pad_y).saturating_sub(h);
                let hi = by1.saturating_sub(pad_y).min(gh - h);
                let (y_lo, y_hi) = if lo <= hi { (lo, hi) } else { (0, gh - h) };
                (x_lo, x_hi, y_lo, y_hi)
            }
        };
        for y in y_lo..=y_hi {
            for x in x_lo..=x_hi {
                let s = score(x, y, w, h);
                let better = match best {
                    None => true,
                    Some((bs, _, _, bw, bh)) => {
                        s > bs
                            || (s == bs
                                && match mode {
                                    SatCropMode::Minimal => w * h > bw * bh,
                                    SatCropMode::Maximal => w * h < bw * bh,
                                })
                    }
                };
                if better {
                    best = Some((s, x, y, w, h));
                }
            }
        }
    }

    let (_, gx1, gy1, cw, chh) = best?;
    Some(grid_rect_to_src(
        gx1,
        gy1,
        gx1 + cw,
        gy1 + chh,
        to_src,
        target_aspect,
    ))
}

/// Map a winning cell rect to exact source pixels at the requested aspect.
fn grid_rect_to_src(
    gx1: usize,
    gy1: usize,
    gx2: usize,
    gy2: usize,
    m: &TargetToSrc,
    (aw, ah): (u32, u32),
) -> CropRect {
    // Cell edges → target px (×8) → source px.
    let sx1 = m.map_x(gx1 as f64 * 8.0);
    let sx2 = m.map_x(gx2 as f64 * 8.0);
    let sy1 = m.map_y(gy1 as f64 * 8.0);
    let sy2 = m.map_y(gy2 as f64 * 8.0);
    let (cx, cy) = ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0);
    let (mut w, mut h) = (sx2 - sx1, sy2 - sy1);

    // Shrink the over-long dimension to restore the exact aspect.
    let need_h = w * ah as f64 / aw as f64;
    if need_h <= h {
        h = need_h;
    } else {
        w = h * aw as f64 / ah as f64;
    }
    let w_px = (round64(w) as u32).clamp(1, m.src_w);
    let h_px = (round64(h) as u32).clamp(1, m.src_h);
    let x = (round64(cx - w / 2.0) as i64).clamp(0, (m.src_w - w_px) as i64) as u32;
    let y = (round64(cy - h / 2.0) as i64).clamp(0, (m.src_h - h_px) as i64) as u32;
    CropRect {
        x,
        y,
        w: w_px,
        h: h_px,
    }
}

// ---------------------------------------------------------------------------
// find_distant_subject_tile — Tile-on-Demand trigger
// ---------------------------------------------------------------------------

/// Knobs for [`find_distant_subject_tile`].
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Minimum peak height (of `max(matte, body)`) to count as a subject
    /// (default 0.3).
    pub subject_threshold: f32,
    /// Face-channel value that counts as "Face fired" inside the blob
    /// (default 0.25).
    pub face_threshold: f32,
    /// Maximum fraction of the grid a compact subject may cover
    /// (default 0.04).
    pub max_area_frac: f32,
    /// Tile edge in native source pixels (default 256).
    pub tile_px: u32,
    /// Vertical placement: rect center lands at
    /// `blob_top + head_frac * blob_h` — the subject's upper/head region
    /// (default 0.3).
    pub head_frac: f32,
    /// Radius in cells scanned around each peak when measuring compactness
    /// (default 12).
    pub window_radius: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            subject_threshold: 0.3,
            face_threshold: 0.25,
            max_area_frac: 0.04,
            tile_px: 256,
            head_frac: 0.3,
            window_radius: 12,
        }
    }
}

/// Tile-on-Demand trigger.
///
/// Finds compact human/subject peaks (local maxima of `max(matte, body)`)
/// covering less than `max_area_frac` of the frame where the `face` channel
/// did not fire — the "distant person" case the full-frame pass misses —
/// and returns a `tile_px`-square native-resolution [`CropRect`] centered
/// horizontally on the peak blob and vertically on its upper/head region.
///
/// Returns `None` when no qualifying peak exists (no subject, a face
/// already fired there, or the peak covers too much of the frame).
pub fn find_distant_subject_tile(
    matte: &[f32],
    body: &[f32],
    face: &[f32],
    grid_w: usize,
    grid_h: usize,
    to_src: &TargetToSrc,
    cfg: &TileConfig,
) -> Option<CropRect> {
    let (gw, gh) = (grid_w, grid_h);
    let n = gw * gh;
    assert_eq!(matte.len(), n, "matte len != grid");
    assert_eq!(body.len(), n, "body len != grid");
    assert_eq!(face.len(), n, "face len != grid");
    if n == 0 {
        return None;
    }
    let subj = |x: usize, y: usize| matte[y * gw + x].max(body[y * gw + x]);

    // Top-8 local maxima (value-descending) — bounded, allocation-free.
    const MAX_PEAKS: usize = 8;
    let mut peaks = [(0.0f32, 0usize, 0usize); MAX_PEAKS];
    let mut npeaks = 0usize;
    for y in 0..gh {
        for x in 0..gw {
            let v = subj(x, y);
            if v < cfg.subject_threshold {
                continue;
            }
            let mut is_peak = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0
                        && ny >= 0
                        && (nx as usize) < gw
                        && (ny as usize) < gh
                        && subj(nx as usize, ny as usize) > v
                    {
                        is_peak = false;
                    }
                }
            }
            if !is_peak {
                continue;
            }
            if npeaks < MAX_PEAKS {
                let mut i = npeaks;
                npeaks += 1;
                while i > 0 && peaks[i - 1].0 < v {
                    peaks[i] = peaks[i - 1];
                    i -= 1;
                }
                peaks[i] = (v, x, y);
            } else if v > peaks[MAX_PEAKS - 1].0 {
                let mut i = MAX_PEAKS - 1;
                while i > 0 && peaks[i - 1].0 < v {
                    peaks[i] = peaks[i - 1];
                    i -= 1;
                }
                peaks[i] = (v, x, y);
            }
        }
    }

    let r = cfg.window_radius;
    for &(v, px, py) in &peaks[..npeaks] {
        // Compactness gate is global: a window-local count would undercount
        // blobs clipped by the window edge (e.g. a frame-filling matte at a
        // corner peak).
        let mut count = 0usize;
        for i in 0..n {
            if matte[i].max(body[i]) >= 0.5 * v {
                count += 1;
            }
        }
        if count as f32 / n as f32 >= cfg.max_area_frac {
            continue; // Not compact — not a distant subject.
        }
        // Blob bbox and face check stay window-local: they describe *this*
        // peak, not a second subject elsewhere in the frame.
        let x_lo = px.saturating_sub(r);
        let x_hi = (px + r).min(gw - 1);
        let y_lo = py.saturating_sub(r);
        let y_hi = (py + r).min(gh - 1);
        let (mut bx1, mut by1, mut bx2, mut by2) = (px, py, px + 1, py + 1);
        let mut face_max = 0.0f32;
        for y in y_lo..=y_hi {
            for x in x_lo..=x_hi {
                face_max = face_max.max(face[y * gw + x]);
                if subj(x, y) >= 0.5 * v {
                    bx1 = bx1.min(x);
                    by1 = by1.min(y);
                    bx2 = bx2.max(x + 1);
                    by2 = by2.max(y + 1);
                }
            }
        }
        if face_max >= cfg.face_threshold {
            continue; // Face already fired here — the full-frame pass has it.
        }

        // Center x on the blob; center y on its upper/head region.
        let blob_cx = (bx1 + bx2) as f64 / 2.0;
        let head_y = by1 as f64 + (by2 - by1) as f64 * cfg.head_frac as f64;
        let cx_px = to_src.map_x(blob_cx * 8.0);
        let cy_px = to_src.map_y(head_y * 8.0);
        let tw = cfg.tile_px.min(to_src.src_w);
        let th = cfg.tile_px.min(to_src.src_h);
        let x =
            (round64(cx_px - tw as f64 / 2.0) as i64).clamp(0, (to_src.src_w - tw) as i64) as u32;
        let y =
            (round64(cy_px - th as f64 / 2.0) as i64).clamp(0, (to_src.src_h - th) as i64) as u32;
        return Some(CropRect { x, y, w: tw, h: th });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_grid(gw: usize, gh: usize) -> Vec<f32> {
        let mut s = 0x9e37_79b9u32;
        (0..gw * gh)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 24) as f32) / 255.0 * 4.0
            })
            .collect()
    }

    fn naive_sat_sum(cells: &[u32], gw: usize, x1: usize, y1: usize, x2: usize, y2: usize) -> u32 {
        let mut s = 0u64;
        for y in y1..y2 {
            for x in x1..x2 {
                s += cells[y * gw + x] as u64;
            }
        }
        s as u32
    }

    fn map_from(energy: &[f32], cut: &[f32], gw: usize, gh: usize) -> DualIntegralMap {
        DualIntegralMap::build(
            &IntegralChannels {
                saliency: Some(energy),
                cut_penalty: Some(cut),
                ..Default::default()
            },
            gw,
            gh,
            0.25,
        )
    }

    #[test]
    fn sat_exact_vs_naive_all_rects() {
        let (gw, gh) = (9usize, 7usize);
        let e = lcg_grid(gw, gh);
        let c = lcg_grid(gw, gh);
        let map = map_from(&e, &c, gw, gh);
        // Rebuild the clamped fixed-point cells the map actually stores.
        let cap_f = (u32::MAX / (gw * gh) as u32) as f32 / SAT_SCALE;
        let fe: Vec<u32> = e
            .iter()
            .map(|&v| (v.clamp(0.0, cap_f) * SAT_SCALE).round() as u32)
            .collect();
        let fc: Vec<u32> = c
            .iter()
            .map(|&v| (v.clamp(0.0, cap_f) * SAT_SCALE).round() as u32)
            .collect();
        for x1 in 0..gw {
            for x2 in x1 + 1..=gw {
                for y1 in 0..gh {
                    for y2 in y1 + 1..=gh {
                        assert_eq!(
                            map.rect_energy(x1, y1, x2, y2),
                            naive_sat_sum(&fe, gw, x1, y1, x2, y2),
                            "energy rect ({x1},{y1})-({x2},{y2})"
                        );
                        for band in 1..=2usize {
                            let yt = (y1 + band).min(y2);
                            let yb = y2.saturating_sub(band).max(yt);
                            let xl = (x1 + band).min(x2);
                            let xr = x2.saturating_sub(band).max(xl);
                            let want = naive_sat_sum(&fc, gw, x1, y1, x2, yt)
                                + naive_sat_sum(&fc, gw, x1, yb, x2, y2)
                                + naive_sat_sum(&fc, gw, x1, yt, xl, yb)
                                + naive_sat_sum(&fc, gw, xr, yt, x2, yb);
                            assert_eq!(
                                map.rect_edge_cut_penalty(x1, y1, x2, y2, band),
                                want,
                                "band {band} edge ({x1},{y1})-({x2},{y2})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn minimal_returns_largest_on_flat_energy() {
        let (gw, gh) = (16usize, 12usize);
        let e = alloc::vec![1.0f32; gw * gh];
        let z = alloc::vec![0.0f32; gw * gh];
        let map = map_from(&e, &z, gw, gh);
        // stretch 640×480 → 128×96: cell = 8×5 = 40 src px square.
        let ts = TargetToSrc::stretch(640, 480, 128, 96);
        let r = solve_crop_sat(
            &map,
            &ts,
            (1, 1),
            SatCropMode::Minimal,
            &SatCropConfig::default(),
        )
        .unwrap();
        // Largest 1:1 at 4:3 grid → 12×12 cells → 480×480 source px.
        assert_eq!((r.w, r.h), (480, 480));
        assert!(r.x + r.w <= 640 && r.y + r.h <= 480);
    }

    #[test]
    fn minimal_dodges_eye_chin_cut_band() {
        // Vertical "face strip" of high cut penalty at cols 3..5, full
        // height — a crop edge landing inside it means slicing the subject.
        // Energy peaks at cols >= 12, so the naive best-energy placement
        // (x=4) puts its left edge inside the band; valid placements exist
        // at x <= 2, and the solver must find one.
        let (gw, gh) = (16usize, 12usize);
        let mut e = alloc::vec![1.0f32; gw * gh];
        for y in 0..gh {
            for x in 12..gw {
                e[y * gw + x] = 4.0;
            }
        }
        let mut c = alloc::vec![0.0f32; gw * gh];
        for y in 0..gh {
            for x in 3..5 {
                c[y * gw + x] = 50.0;
            }
        }
        let map = map_from(&e, &c, gw, gh);
        let ts = TargetToSrc::stretch(640, 480, 128, 96);
        let cfg = SatCropConfig::default();
        let r = solve_crop_sat(&map, &ts, (1, 1), SatCropMode::Minimal, &cfg).unwrap();
        // The naive best-energy full-height placement (12×12 @ x=4) puts its
        // left edge inside the band — show it really is penalized, then
        // verify the winner's border is entirely outside it.
        assert!(map.rect_edge_cut_penalty(4, 0, 16, 12, 1) > 0);
        let gx = (r.x as f64 / (8.0 * ts.x.scale)).round() as usize;
        let gy = (r.y as f64 / (8.0 * ts.y.scale)).round() as usize;
        let gw2 = (r.w as f64 / (8.0 * ts.x.scale)).round() as usize;
        let gh2 = (r.h as f64 / (8.0 * ts.y.scale)).round() as usize;
        assert_eq!(
            map.rect_edge_cut_penalty(gx, gy, gx + gw2, gy + gh2, 1),
            0,
            "winner border crosses the band: {r:?}"
        );
    }

    #[test]
    fn maximal_tighter_than_minimal_around_matte() {
        let (gw, gh) = (16usize, 12usize);
        let mut matte = alloc::vec![0.0f32; gw * gh];
        for y in 4..8 {
            for x in 6..10 {
                matte[y * gw + x] = 0.9;
            }
        }
        let z = alloc::vec![0.0f32; gw * gh];
        let map = DualIntegralMap::build(
            &IntegralChannels {
                matte: Some(&matte),
                cut_penalty: Some(&z),
                ..Default::default()
            },
            gw,
            gh,
            0.25,
        );
        let ts = TargetToSrc::stretch(640, 480, 128, 96);
        let cfg = SatCropConfig::default();
        let min = solve_crop_sat(&map, &ts, (1, 1), SatCropMode::Minimal, &cfg).unwrap();
        let max = solve_crop_sat(&map, &ts, (1, 1), SatCropMode::Maximal, &cfg).unwrap();
        assert!(
            max.w < min.w || max.h < min.h,
            "Maximal {max:?} not tighter than Minimal {min:?}"
        );
        // Maximal still wraps the subject (blob center stays inside).
        let bx = 8.0 * 8.0 * ts.x.scale;
        let by = 6.0 * 8.0 * ts.y.scale;
        assert!(
            (bx as u32) >= max.x && (bx as u32) <= max.x + max.w,
            "subject center x escapes Maximal crop"
        );
        assert!(
            (by as u32) >= max.y && (by as u32) <= max.y + max.h,
            "subject center y escapes Maximal crop"
        );
    }

    #[test]
    fn distant_subject_tile_fires() {
        // 3840×2160 source → 640×480 target → 80×60 grid (stretch 6, 4.5).
        let (gw, gh) = (80usize, 60usize);
        let mut matte = alloc::vec![0.0f32; gw * gh];
        let mut body = alloc::vec![0.0f32; gw * gh];
        let face = alloc::vec![0.0f32; gw * gh];
        // Compact torso blob at cells (60..62) x (24..27): ~0.13% of frame.
        for y in 24..27 {
            for x in 60..62 {
                body[y * gw + x] = 0.6;
            }
        }
        body[25 * gw + 60] = 0.9;
        matte[25 * gw + 60] = 0.4;
        let ts = TargetToSrc::stretch(3840, 2160, 640, 480);
        let cfg = TileConfig::default();
        let r = find_distant_subject_tile(&matte, &body, &face, gw, gh, &ts, &cfg).unwrap();
        assert_eq!((r.w, r.h), (256, 256));
        // Blob bbox x = (60,62): center 61 cells → src 61*8*6 = 2928 → x = 2800.
        let want_x = (61.0 * 8.0 * ts.x.scale - 128.0).round() as u32;
        assert_eq!(r.x, want_x);
        // Vertically on the blob's upper region: head_y = 24 + 3*0.3 = 24.9
        // cells → 24.9*8*4.5 = 896.4 → y = 768.
        let want_y =
            ((24.0 + 3.0 * cfg.head_frac as f64) * 8.0 * ts.y.scale - 128.0).round() as u32;
        assert_eq!(r.y, want_y);
        assert!(r.x + r.w <= 3840 && r.y + r.h <= 2160);
    }

    #[test]
    fn distant_subject_tile_suppressed_by_face() {
        let (gw, gh) = (80usize, 60usize);
        let mut body = alloc::vec![0.0f32; gw * gh];
        let mut face = alloc::vec![0.0f32; gw * gh];
        let matte = alloc::vec![0.0f32; gw * gh];
        for y in 24..27 {
            for x in 60..62 {
                body[y * gw + x] = 0.8;
            }
        }
        face[25 * gw + 60] = 0.6; // Face fired near the peak.
        let ts = TargetToSrc::stretch(3840, 2160, 640, 480);
        assert!(
            find_distant_subject_tile(&matte, &body, &face, gw, gh, &ts, &TileConfig::default())
                .is_none()
        );
    }

    #[test]
    fn distant_subject_tile_suppressed_by_size() {
        let (gw, gh) = (80usize, 60usize);
        let matte = alloc::vec![1.0f32; gw * gh]; // subject fills the frame
        let body = alloc::vec![0.9f32; gw * gh];
        let face = alloc::vec![0.0f32; gw * gh];
        let ts = TargetToSrc::stretch(3840, 2160, 640, 480);
        assert!(
            find_distant_subject_tile(&matte, &body, &face, gw, gh, &ts, &TileConfig::default())
                .is_none()
        );
    }

    #[test]
    fn from_crop_inputs_protects_face_and_landmarks() {
        let (gw, gh) = (64usize, 48usize);
        // 512×384 stretch: cell = 8 src px; a 64×48 map aligns 1:1 with cells.
        let ts = TargetToSrc::stretch(512, 384, 512, 384);
        // Modest saliency under the face+torso region anchors the subject.
        let mut sal_data = alloc::vec![0.0f32; gw * gh];
        for y in 10..32 {
            for x in 20..28 {
                sal_data[y * gw + x] = 0.5;
            }
        }
        let sal = SaliencyMap {
            data: sal_data,
            width: 64,
            height: 48,
        };
        // Face at src px (160..224)×(80..144) = cells (20..28)×(10..18),
        // with 5-point landmarks in the eye..mouth interior.
        let face = FaceLandmarkInput {
            rect: FaceRect {
                x1: 31.25,
                y1: 80.0 / 384.0 * 100.0,
                x2: 43.75,
                y2: 37.5,
                confidence: 0.9,
            },
            landmarks: Some([
                [37.5, 104.0 / 384.0 * 100.0],
                [40.625, 104.0 / 384.0 * 100.0],
                [39.0625, 31.25],
                [37.5, 34.375],
                [40.625, 34.375],
            ]),
        };
        let faces = [face];
        let sources = CropInputSources {
            saliency: Some(&sal),
            faces: &faces,
            ..Default::default()
        };
        let map = DualIntegralMap::from_crop_inputs(&sources, gw, gh, &ts, 0.25);
        assert_eq!(map.faces().len(), 1);
        // High-energy landmark core (eye-to-mouth region ≈ cells 24..27 ×
        // 13..17): 6*conf face + 10*conf landmark + saliency ≈ 14+/cell.
        let core_e = map.rect_energy(24, 13, 27, 17) as f32 / SAT_SCALE;
        assert!(core_e > 60.0, "weak landmark core energy {core_e}");
        // Eye-to-chin cut penalties are live under the band.
        assert!(map.rect_cut(24, 13, 27, 17) > 0);
        // Zero face cuts: every aspect × mode contains the face in src px.
        let (fx1, fy1, fx2, fy2) = (160u32, 80u32, 224u32, 144u32);
        let cfg = SatCropConfig::default();
        for &(aw, ah) in &[(1u32, 1u32), (16, 9), (9, 16), (4, 3), (3, 4), (4, 5)] {
            for mode in [SatCropMode::Minimal, SatCropMode::Maximal] {
                let r = solve_crop_sat(&map, &ts, (aw, ah), mode, &cfg).unwrap();
                assert!(
                    r.x <= fx1 && r.y <= fy1 && r.x + r.w >= fx2 && r.y + r.h >= fy2,
                    "{mode:?} {aw}:{ah} crop {r:?} slices or drops the face"
                );
            }
        }
    }

    #[test]
    fn from_crop_inputs_resists_background_speckles_and_classical_noise() {
        let (gw, gh) = (64usize, 48usize);
        let ts = TargetToSrc::stretch(512, 384, 512, 384);
        let mut sal_data = alloc::vec![0.0f32; gw * gh];
        // Compact salient subject at cells (30..38)×(20..30).
        for y in 20..30 {
            for x in 30..38 {
                sal_data[y * gw + x] = 0.9;
            }
        }
        // Isolated hot speckle in the far corner + saturating classical noise.
        sal_data[47 * gw + 63] = 1.0;
        let sal = SaliencyMap {
            data: sal_data,
            width: 64,
            height: 48,
        };
        let classical = alloc::vec![1.0f32; gw * gh];
        let sources = CropInputSources {
            saliency: Some(&sal),
            classical_energy: Some(&classical),
            ..Default::default()
        };
        let map = DualIntegralMap::from_crop_inputs(&sources, gw, gh, &ts, 0.25);
        // The 3×3-smoothed subject bbox hugs the blob and never reaches the
        // speckle corner.
        let (sx1, sy1, sx2, sy2) = map.subject_bbox().unwrap();
        assert!(
            (28..=32).contains(&sx1) && (18..=22).contains(&sy1),
            "bbox ({sx1},{sy1},{sx2},{sy2}) pulled off the blob"
        );
        assert!(
            sx2 <= 40 && sy2 <= 32,
            "bbox ({sx1},{sy1},{sx2},{sy2}) stretched toward speckle"
        );
        // Maximal stays on the subject: the speckle's source pixels
        // (504..512, 376..384) are outside and the blob center is inside.
        let r = solve_crop_sat(
            &map,
            &ts,
            (9, 16),
            SatCropMode::Maximal,
            &SatCropConfig::default(),
        )
        .unwrap();
        assert!(
            r.x + r.w <= 504 || r.y + r.h <= 376,
            "Maximal {r:?} reached the speckle corner"
        );
        assert!(
            r.x <= 272 && r.x + r.w >= 280 && r.y <= 200 && r.y + r.h >= 208,
            "Maximal {r:?} misses subject center"
        );
    }

    #[test]
    fn maximal_narrow_aspect_on_wide_subject_zooms_in() {
        let (gw, gh) = (64usize, 48usize);
        // Wide subject: matte blob 40×20 cells at (12..52)×(14..34) with a
        // hotter core the crop must capture.
        let mut matte_data = alloc::vec![0.0f32; gw * gh];
        for y in 14..34 {
            for x in 12..52 {
                matte_data[y * gw + x] = 0.7;
            }
        }
        for y in 20..28 {
            for x in 28..36 {
                matte_data[y * gw + x] = 1.0;
            }
        }
        let matte = SaliencyMap {
            data: matte_data,
            width: 64,
            height: 48,
        };
        let ts = TargetToSrc::stretch(640, 480, 512, 384);
        let sources = CropInputSources {
            matte: Some(&matte),
            ..Default::default()
        };
        let map = DualIntegralMap::from_crop_inputs(&sources, gw, gh, &ts, 0.25);
        let cfg = SatCropConfig::default();
        let min = solve_crop_sat(&map, &ts, (9, 16), SatCropMode::Minimal, &cfg).unwrap();
        let max = solve_crop_sat(&map, &ts, (9, 16), SatCropMode::Maximal, &cfg).unwrap();
        // Pre-fix this collapsed to the full-frame size; Maximal must now be
        // meaningfully tighter than Minimal at the narrow aspect.
        let (a_min, a_max) = (min.w as u64 * min.h as u64, max.w as u64 * max.h as u64);
        assert!(
            a_max * 100 <= 85 * a_min,
            "Maximal {max:?} not ≤ 0.85× Minimal {min:?}"
        );
        // Peak cell (32,24) → src px (320..328)×(240..248) stays inside.
        assert!(
            max.x <= 320 && max.x + max.w >= 328 && max.y <= 240 && max.y + max.h >= 248,
            "Maximal {max:?} misses subject peak"
        );
    }
}
