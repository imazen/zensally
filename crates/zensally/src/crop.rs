//! Smart portrait crop heuristic combining face detection and saliency.
//!
//! Two modes per aspect ratio:
//! - **Minimal**: largest crop at target ratio, positioned to keep faces/saliency visible.
//! - **Maximal**: tightest crop at target ratio, zoomed in on the subject.

use crate::{FaceRect, SaliencyMap};

/// An integer crop rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Target aspect ratio as integer width:height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspectRatio {
    pub w: u32,
    pub h: u32,
}

pub const PORTRAIT_9_16: AspectRatio = AspectRatio { w: 9, h: 16 };
pub const PORTRAIT_3_4: AspectRatio = AspectRatio { w: 3, h: 4 };
pub const PORTRAIT_4_5: AspectRatio = AspectRatio { w: 4, h: 5 };
pub const SQUARE: AspectRatio = AspectRatio { w: 1, h: 1 };
pub const LANDSCAPE_16_9: AspectRatio = AspectRatio { w: 16, h: 9 };
pub const LANDSCAPE_4_3: AspectRatio = AspectRatio { w: 4, h: 3 };

/// Crop strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropMode {
    /// Largest crop at target ratio. Removes minimum content.
    Minimal,
    /// Tightest crop at target ratio. Zooms in on the subject.
    Maximal,
}

/// Configuration for [`compute_crop`].
#[derive(Debug, Clone)]
pub struct CropConfig {
    /// Target aspect ratio (default: 9:16 portrait).
    pub target_aspect: AspectRatio,
    /// Crop strategy (default: Minimal).
    pub mode: CropMode,
    /// Where to place the primary face center vertically within the crop,
    /// as a fraction from the top (default: 0.38 — eyes land near the top third).
    pub face_vertical_position: f32,
    /// Minimum fraction of each face's area that must remain inside the crop
    /// (minimal mode, default: 0.7).
    pub min_face_visibility: f32,
    /// Padding around the subject as a fraction of subject size
    /// (maximal mode, default: 0.5).
    pub zoom_padding: f32,
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            target_aspect: PORTRAIT_9_16,
            mode: CropMode::Minimal,
            face_vertical_position: 0.38,
            min_face_visibility: 0.7,
            zoom_padding: 0.5,
        }
    }
}

/// Compute the optimal crop rectangle for the given source image.
///
/// Returns `None` if the source dimensions are degenerate (zero width or height).
pub fn compute_crop(
    src_w: u32,
    src_h: u32,
    faces: &[FaceRect],
    saliency: Option<&SaliencyMap>,
    config: &CropConfig,
) -> Option<CropRect> {
    if src_w == 0 || src_h == 0 || config.target_aspect.w == 0 || config.target_aspect.h == 0 {
        return None;
    }

    let qualifying_faces: Vec<&FaceRect> = faces.iter().filter(|f| f.confidence >= 0.5).collect();

    match config.mode {
        CropMode::Minimal => minimal_crop(src_w, src_h, &qualifying_faces, saliency, config),
        CropMode::Maximal => maximal_crop(src_w, src_h, &qualifying_faces, saliency, config),
    }
}

// ---------------------------------------------------------------------------
// Minimal mode
// ---------------------------------------------------------------------------

fn minimal_crop(
    src_w: u32,
    src_h: u32,
    faces: &[&FaceRect],
    saliency: Option<&SaliencyMap>,
    config: &CropConfig,
) -> Option<CropRect> {
    let (crop_w, crop_h) = largest_rect_at_ratio(src_w, src_h, config.target_aspect);
    if crop_w == 0 || crop_h == 0 {
        return None;
    }

    let sw = src_w as f64;
    let sh = src_h as f64;
    let cw = crop_w as f64;
    let ch = crop_h as f64;

    let (focus_x, focus_y, has_faces) = find_focus(faces, saliency, sw, sh);

    // Position crop
    let cx = focus_x - cw / 2.0;

    let cy = if has_faces {
        // Place primary face center at face_vertical_position from crop top
        let primary = primary_face(faces);
        let pcy = face_center_y(primary, sh);
        pcy - ch * config.face_vertical_position as f64
    } else {
        focus_y - ch / 2.0
    };

    let mut x = clamp_f64(cx, 0.0, sw - cw);
    let mut y = clamp_f64(cy, 0.0, sh - ch);

    if has_faces {
        // Face visibility check — shift toward clipped faces
        let primary = primary_face(faces);
        let geom = CropGeom { cw, ch, sw, sh };
        shift_for_face_visibility(
            &mut x,
            &mut y,
            &geom,
            faces,
            primary,
            config.min_face_visibility,
        );
    } else if let Some(sal) = saliency {
        // Saliency coverage check — shift to include the salient region,
        // biased toward the top (where the "face" of a subject usually is).
        shift_for_saliency_coverage(&mut x, &mut y, cw, ch, sw, sh, sal);
    }

    Some(CropRect {
        x: x.round() as u32,
        y: y.round() as u32,
        w: crop_w,
        h: crop_h,
    })
}

// ---------------------------------------------------------------------------
// Maximal mode
// ---------------------------------------------------------------------------

fn maximal_crop(
    src_w: u32,
    src_h: u32,
    faces: &[&FaceRect],
    saliency: Option<&SaliencyMap>,
    config: &CropConfig,
) -> Option<CropRect> {
    let sw = src_w as f64;
    let sh = src_h as f64;

    // Step 1: determine subject region (pixel bbox)
    let (mut sx1, mut sy1, mut sx2, mut sy2, has_faces) =
        subject_region(faces, saliency, sw, sh, config.zoom_padding);

    // Step 2: expand to target aspect ratio
    expand_to_aspect(&mut sx1, &mut sy1, &mut sx2, &mut sy2, config.target_aspect);

    // Step 3: enforce minimum size (30% of source on each axis)
    let min_w = sw * 0.3;
    let min_h = sh * 0.3;
    let cur_w = sx2 - sx1;
    let cur_h = sy2 - sy1;
    if cur_w < min_w || cur_h < min_h {
        let scale = f64::max(min_w / cur_w, min_h / cur_h);
        let new_w = cur_w * scale;
        let new_h = cur_h * scale;
        let mid_x = (sx1 + sx2) / 2.0;
        let mid_y = (sy1 + sy2) / 2.0;
        sx1 = mid_x - new_w / 2.0;
        sy1 = mid_y - new_h / 2.0;
        sx2 = mid_x + new_w / 2.0;
        sy2 = mid_y + new_h / 2.0;
        // Re-expand to aspect ratio after scaling
        expand_to_aspect(&mut sx1, &mut sy1, &mut sx2, &mut sy2, config.target_aspect);
    }

    // Step 4: headroom adjustment (face mode)
    if has_faces {
        let primary = primary_face(faces);
        let pcy = face_center_y(primary, sh);
        let crop_h = sy2 - sy1;
        let desired_top = pcy - crop_h * config.face_vertical_position as f64;
        let shift = desired_top - sy1;
        sy1 += shift;
        sy2 += shift;
    }

    // Step 5: clamp to image bounds
    clamp_rect_to_bounds(&mut sx1, &mut sy1, &mut sx2, &mut sy2, sw, sh);

    // Step 6: enforce aspect ratio after clamping (clamping may have shrunk one axis)
    enforce_aspect_after_clamp(&mut sx1, &mut sy1, &mut sx2, &mut sy2, sw, sh, config.target_aspect);

    let crop_w = (sx2 - sx1).round() as u32;
    let crop_h = (sy2 - sy1).round() as u32;
    if crop_w == 0 || crop_h == 0 {
        return None;
    }

    Some(CropRect {
        x: sx1.round() as u32,
        y: sy1.round() as u32,
        w: crop_w,
        h: crop_h,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Largest rectangle at the given aspect ratio that fits within src dimensions.
fn largest_rect_at_ratio(src_w: u32, src_h: u32, ratio: AspectRatio) -> (u32, u32) {
    let rw = ratio.w as f64;
    let rh = ratio.h as f64;
    let sw = src_w as f64;
    let sh = src_h as f64;

    // Try width-limited: w = src_w, h = src_w * rh / rw
    let h_if_w = sw * rh / rw;
    if h_if_w <= sh {
        (src_w, h_if_w.floor() as u32)
    } else {
        // Height-limited: h = src_h, w = src_h * rw / rh
        let w_if_h = sh * rw / rh;
        (w_if_h.floor() as u32, src_h)
    }
}

/// Find the focus point for positioning the crop.
/// Returns (focus_x, focus_y, has_faces).
fn find_focus(
    faces: &[&FaceRect],
    saliency: Option<&SaliencyMap>,
    sw: f64,
    sh: f64,
) -> (f64, f64, bool) {
    if !faces.is_empty() {
        // Face mode: focus_x = center of bounding box enclosing ALL faces,
        //            focus_y = primary face center
        let (all_x1, _, all_x2, _) = enclosing_bbox_pixels(faces, sw, sh);
        let focus_x = (all_x1 + all_x2) / 2.0;
        let primary = primary_face(faces);
        let focus_y = face_center_y(primary, sh);
        (focus_x, focus_y, true)
    } else if let Some(sal) = saliency {
        let (cx, cy) = saliency_center_of_mass(sal);
        (cx * sw, cy * sh, false)
    } else {
        (sw / 2.0, sh / 2.0, false)
    }
}

/// Determine subject region for maximal mode.
/// Returns (x1, y1, x2, y2, has_faces) in pixel coordinates with padding applied.
fn subject_region(
    faces: &[&FaceRect],
    saliency: Option<&SaliencyMap>,
    sw: f64,
    sh: f64,
    zoom_padding: f32,
) -> (f64, f64, f64, f64, bool) {
    let pad = zoom_padding as f64;

    if !faces.is_empty() {
        let primary = primary_face(faces);

        // Find which faces are "close" to primary (within 2x primary face width)
        let pw = (primary.x2 - primary.x1) as f64 / 100.0 * sw;
        let pcx = (primary.x1 as f64 + primary.x2 as f64) / 2.0 / 100.0 * sw;
        let pcy = (primary.y1 as f64 + primary.y2 as f64) / 2.0 / 100.0 * sh;

        let close_faces: Vec<&&FaceRect> = faces
            .iter()
            .filter(|f| {
                let fcx = (f.x1 as f64 + f.x2 as f64) / 2.0 / 100.0 * sw;
                let fcy = (f.y1 as f64 + f.y2 as f64) / 2.0 / 100.0 * sh;
                let dist =
                    ((fcx - pcx) * (fcx - pcx) + (fcy - pcy) * (fcy - pcy)).sqrt();
                dist <= pw * 2.0
            })
            .collect();

        let close_refs: Vec<&FaceRect> = close_faces.iter().map(|f| **f).collect();
        let (bx1, by1, bx2, by2) = enclosing_bbox_pixels(&close_refs, sw, sh);
        let bw = bx2 - bx1;
        let bh = by2 - by1;

        (
            bx1 - bw * pad,
            by1 - bh * pad,
            bx2 + bw * pad,
            by2 + bh * pad,
            true,
        )
    } else if let Some(sal) = saliency {
        let (sx1, sy1, sx2, sy2) = saliency_bbox(sal, 0.5);
        let bx1 = sx1 * sw;
        let by1 = sy1 * sh;
        let bx2 = sx2 * sw;
        let by2 = sy2 * sh;
        let bw = bx2 - bx1;
        let bh = by2 - by1;
        (
            bx1 - bw * pad,
            by1 - bh * pad,
            bx2 + bw * pad,
            by2 + bh * pad,
            false,
        )
    } else {
        // Fallback: center 50%
        (sw * 0.25, sh * 0.25, sw * 0.75, sh * 0.75, false)
    }
}

/// Primary face = largest by area.
fn primary_face<'a>(faces: &[&'a FaceRect]) -> &'a FaceRect {
    faces
        .iter()
        .max_by(|a, b| {
            let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
            let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
            area_a.partial_cmp(&area_b).unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap()
}

/// Face center Y in pixel coordinates.
fn face_center_y(face: &FaceRect, src_h: f64) -> f64 {
    (face.y1 as f64 + face.y2 as f64) / 2.0 / 100.0 * src_h
}

/// Bounding box enclosing all faces, in pixel coordinates.
fn enclosing_bbox_pixels(faces: &[&FaceRect], sw: f64, sh: f64) -> (f64, f64, f64, f64) {
    let mut x1 = f64::MAX;
    let mut y1 = f64::MAX;
    let mut x2 = f64::MIN;
    let mut y2 = f64::MIN;
    for f in faces {
        x1 = x1.min(f.x1 as f64 / 100.0 * sw);
        y1 = y1.min(f.y1 as f64 / 100.0 * sh);
        x2 = x2.max(f.x2 as f64 / 100.0 * sw);
        y2 = y2.max(f.y2 as f64 / 100.0 * sh);
    }
    (x1, y1, x2, y2)
}

/// Square-weighted center of mass of saliency map (threshold 0.3).
/// Returns (cx, cy) normalized to [0, 1].
fn saliency_center_of_mass(sal: &SaliencyMap) -> (f64, f64) {
    let mut sum_wx = 0.0_f64;
    let mut sum_wy = 0.0_f64;
    let mut sum_w = 0.0_f64;

    for row in 0..sal.height {
        for col in 0..sal.width {
            let v = sal.data[(row * sal.width + col) as usize] as f64;
            if v < 0.3 {
                continue;
            }
            let w = v * v; // square weighting
            sum_wx += (col as f64 + 0.5) * w;
            sum_wy += (row as f64 + 0.5) * w;
            sum_w += w;
        }
    }

    if sum_w < 1e-10 {
        (0.5, 0.5)
    } else {
        (
            sum_wx / sum_w / sal.width as f64,
            sum_wy / sum_w / sal.height as f64,
        )
    }
}

/// Bounding box of saliency pixels above `threshold` fraction of max.
/// Returns (x1, y1, x2, y2) normalized to [0, 1].
fn saliency_bbox(sal: &SaliencyMap, threshold: f64) -> (f64, f64, f64, f64) {
    let max_val = sal.data.iter().cloned().fold(0.0_f32, f32::max) as f64;
    if max_val < 1e-10 {
        return (0.25, 0.25, 0.75, 0.75);
    }
    let thresh = max_val * threshold;

    let mut min_col = sal.width;
    let mut min_row = sal.height;
    let mut max_col = 0u32;
    let mut max_row = 0u32;

    for row in 0..sal.height {
        for col in 0..sal.width {
            if sal.data[(row * sal.width + col) as usize] as f64 >= thresh {
                min_col = min_col.min(col);
                min_row = min_row.min(row);
                max_col = max_col.max(col);
                max_row = max_row.max(row);
            }
        }
    }

    if max_col < min_col {
        return (0.25, 0.25, 0.75, 0.75);
    }

    (
        min_col as f64 / sal.width as f64,
        min_row as f64 / sal.height as f64,
        (max_col + 1) as f64 / sal.width as f64,
        (max_row + 1) as f64 / sal.height as f64,
    )
}

/// Expand a rectangle to match the target aspect ratio by growing the shorter dimension.
fn expand_to_aspect(
    x1: &mut f64,
    y1: &mut f64,
    x2: &mut f64,
    y2: &mut f64,
    ratio: AspectRatio,
) {
    let cur_w = *x2 - *x1;
    let cur_h = *y2 - *y1;
    let target_ratio = ratio.w as f64 / ratio.h as f64;
    let cur_ratio = cur_w / cur_h;

    if cur_ratio < target_ratio {
        // Too tall — widen
        let new_w = cur_h * target_ratio;
        let mid_x = (*x1 + *x2) / 2.0;
        *x1 = mid_x - new_w / 2.0;
        *x2 = mid_x + new_w / 2.0;
    } else {
        // Too wide — heighten
        let new_h = cur_w / target_ratio;
        let mid_y = (*y1 + *y2) / 2.0;
        *y1 = mid_y - new_h / 2.0;
        *y2 = mid_y + new_h / 2.0;
    }
}

/// Clamp a rectangle to image bounds, preserving size.
fn clamp_rect_to_bounds(
    x1: &mut f64,
    y1: &mut f64,
    x2: &mut f64,
    y2: &mut f64,
    sw: f64,
    sh: f64,
) {
    let w = *x2 - *x1;
    let h = *y2 - *y1;
    if *x1 < 0.0 {
        *x1 = 0.0;
        *x2 = w.min(sw);
    }
    if *y1 < 0.0 {
        *y1 = 0.0;
        *y2 = h.min(sh);
    }
    if *x2 > sw {
        *x2 = sw;
        *x1 = (sw - w).max(0.0);
    }
    if *y2 > sh {
        *y2 = sh;
        *y1 = (sh - h).max(0.0);
    }
}

/// After clamping to image bounds, the aspect ratio may be wrong if the rect
/// was larger than the image. Shrink the excess dimension to restore the ratio.
fn enforce_aspect_after_clamp(
    x1: &mut f64,
    y1: &mut f64,
    x2: &mut f64,
    y2: &mut f64,
    sw: f64,
    sh: f64,
    ratio: AspectRatio,
) {
    let cur_w = *x2 - *x1;
    let cur_h = *y2 - *y1;
    let target_ratio = ratio.w as f64 / ratio.h as f64;
    let cur_ratio = cur_w / cur_h;
    let tolerance = 0.001;

    if (cur_ratio - target_ratio).abs() < tolerance {
        return;
    }

    if cur_ratio > target_ratio {
        // Too wide — shrink width
        let new_w = cur_h * target_ratio;
        let mid_x = (*x1 + *x2) / 2.0;
        *x1 = mid_x - new_w / 2.0;
        *x2 = mid_x + new_w / 2.0;
    } else {
        // Too tall — shrink height
        let new_h = cur_w / target_ratio;
        let mid_y = (*y1 + *y2) / 2.0;
        *y1 = mid_y - new_h / 2.0;
        *y2 = mid_y + new_h / 2.0;
    }

    // Re-clamp (shrinking shouldn't push out of bounds, but be safe)
    if *x1 < 0.0 {
        *x2 -= *x1;
        *x1 = 0.0;
    }
    if *y1 < 0.0 {
        *y2 -= *y1;
        *y1 = 0.0;
    }
    if *x2 > sw {
        *x1 -= *x2 - sw;
        *x2 = sw;
        *x1 = x1.max(0.0);
    }
    if *y2 > sh {
        *y1 -= *y2 - sh;
        *y2 = sh;
        *y1 = y1.max(0.0);
    }
}

/// Fraction of a face's area that lies inside the crop rectangle.
fn face_overlap_fraction(face: &FaceRect, cx: f64, cy: f64, cw: f64, ch: f64, sw: f64, sh: f64) -> f64 {
    let fx1 = face.x1 as f64 / 100.0 * sw;
    let fy1 = face.y1 as f64 / 100.0 * sh;
    let fx2 = face.x2 as f64 / 100.0 * sw;
    let fy2 = face.y2 as f64 / 100.0 * sh;
    let face_area = (fx2 - fx1) * (fy2 - fy1);
    if face_area < 1e-10 {
        return 1.0;
    }

    let ox1 = fx1.max(cx);
    let oy1 = fy1.max(cy);
    let ox2 = fx2.min(cx + cw);
    let oy2 = fy2.min(cy + ch);
    let overlap = (ox2 - ox1).max(0.0) * (oy2 - oy1).max(0.0);
    overlap / face_area
}

/// Crop geometry for face visibility calculations.
struct CropGeom {
    cw: f64,
    ch: f64,
    sw: f64,
    sh: f64,
}

/// Shift crop position to improve face visibility.
/// Prioritizes primary face if not all faces can be satisfied.
fn shift_for_face_visibility(
    x: &mut f64,
    y: &mut f64,
    geom: &CropGeom,
    faces: &[&FaceRect],
    primary: &FaceRect,
    min_visibility: f32,
) {
    let min_vis = min_visibility as f64;
    let CropGeom { cw, ch, sw, sh } = *geom;

    // Try to include all faces
    for face in faces {
        let frac = face_overlap_fraction(face, *x, *y, cw, ch, sw, sh);
        if frac >= min_vis {
            continue;
        }

        let fx1 = face.x1 as f64 / 100.0 * sw;
        let fy1 = face.y1 as f64 / 100.0 * sh;
        let fx2 = face.x2 as f64 / 100.0 * sw;
        let fy2 = face.y2 as f64 / 100.0 * sh;

        // Shift toward the clipped face
        if fx1 < *x {
            *x = clamp_f64(fx1, 0.0, sw - cw);
        }
        if fx2 > *x + cw {
            *x = clamp_f64(fx2 - cw, 0.0, sw - cw);
        }
        if fy1 < *y {
            *y = clamp_f64(fy1, 0.0, sh - ch);
        }
        if fy2 > *y + ch {
            *y = clamp_f64(fy2 - ch, 0.0, sh - ch);
        }
    }

    // Ensure primary face still meets threshold after shifts for others
    let primary_frac = face_overlap_fraction(primary, *x, *y, cw, ch, sw, sh);
    if primary_frac < min_vis {
        let fx1 = primary.x1 as f64 / 100.0 * sw;
        let fy1 = primary.y1 as f64 / 100.0 * sh;
        let fx2 = primary.x2 as f64 / 100.0 * sw;
        let fy2 = primary.y2 as f64 / 100.0 * sh;
        let fcx = (fx1 + fx2) / 2.0;
        let fcy = (fy1 + fy2) / 2.0;
        *x = clamp_f64(fcx - cw / 2.0, 0.0, sw - cw);
        *y = clamp_f64(fcy - ch / 2.0, 0.0, sh - ch);
    }
}

/// Shift crop to maximize coverage of the salient region.
///
/// The saliency CoM gives a good center, but when the crop is much smaller
/// than the image (e.g. 16:9 from a tall portrait), centering on the CoM
/// can clip the top of the subject (head/face). This shifts the crop to
/// cover as much of the salient bbox as possible, biased toward the top.
fn shift_for_saliency_coverage(
    x: &mut f64,
    y: &mut f64,
    cw: f64,
    ch: f64,
    sw: f64,
    sh: f64,
    sal: &SaliencyMap,
) {
    let (bx1, by1, bx2, by2) = saliency_bbox(sal, 0.3);
    let sal_x1 = bx1 * sw;
    let sal_y1 = by1 * sh;
    let sal_x2 = bx2 * sw;
    let sal_y2 = by2 * sh;

    // If the salient region top is above the crop, shift up to include it.
    // If the salient region bottom is below the crop, shift down.
    // Prioritize showing the top of the salient region (where the "face" is).
    if sal_y1 < *y && sal_y2 > *y + ch {
        // Salient region is taller than the crop — prioritize the top
        *y = clamp_f64(sal_y1, 0.0, sh - ch);
    } else if sal_y1 < *y {
        *y = clamp_f64(sal_y1, 0.0, sh - ch);
    } else if sal_y2 > *y + ch {
        *y = clamp_f64(sal_y2 - ch, 0.0, sh - ch);
    }

    // Same for horizontal
    if sal_x1 < *x && sal_x2 > *x + cw {
        // Center on the salient region
        let sal_cx = (sal_x1 + sal_x2) / 2.0;
        *x = clamp_f64(sal_cx - cw / 2.0, 0.0, sw - cw);
    } else if sal_x1 < *x {
        *x = clamp_f64(sal_x1, 0.0, sw - cw);
    } else if sal_x2 > *x + cw {
        *x = clamp_f64(sal_x2 - cw, 0.0, sw - cw);
    }
}

fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> FaceRect {
        FaceRect {
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }

    fn make_saliency(width: u32, height: u32, hot_spots: &[(u32, u32, f32)]) -> SaliencyMap {
        let mut data = vec![0.0f32; (width * height) as usize];
        for &(col, row, val) in hot_spots {
            if col < width && row < height {
                data[(row * width + col) as usize] = val;
            }
        }
        SaliencyMap {
            data,
            width,
            height,
        }
    }

    /// Fill a rectangular region in the saliency map.
    fn make_saliency_rect(
        width: u32,
        height: u32,
        rx1: u32,
        ry1: u32,
        rx2: u32,
        ry2: u32,
        val: f32,
    ) -> SaliencyMap {
        let mut data = vec![0.0f32; (width * height) as usize];
        for row in ry1..ry2.min(height) {
            for col in rx1..rx2.min(width) {
                data[(row * width + col) as usize] = val;
            }
        }
        SaliencyMap {
            data,
            width,
            height,
        }
    }

    fn assert_crop_inside(crop: &CropRect, src_w: u32, src_h: u32) {
        assert!(
            crop.x + crop.w <= src_w,
            "crop right edge {} exceeds src width {}",
            crop.x + crop.w,
            src_w
        );
        assert!(
            crop.y + crop.h <= src_h,
            "crop bottom edge {} exceeds src height {}",
            crop.y + crop.h,
            src_h
        );
    }

    fn assert_approx_aspect(crop: &CropRect, ratio: AspectRatio, tolerance: f64) {
        let actual = crop.w as f64 / crop.h as f64;
        let expected = ratio.w as f64 / ratio.h as f64;
        let diff = (actual - expected).abs();
        assert!(
            diff < tolerance,
            "aspect {actual:.4} not within {tolerance} of expected {expected:.4} (crop {}x{})",
            crop.w,
            crop.h
        );
    }

    // 1. Landscape 1920x1080 → 9:16, centered face → ~608x1080, face in upper third
    #[test]
    fn minimal_landscape_centered_face() {
        let faces = [face(40.0, 30.0, 60.0, 60.0, 0.9)];
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        assert_eq!(crop.h, 1080); // full height used
        // Face center Y = 45% of 1080 = 486px
        // With face_vertical_position=0.38: crop_top = 486 - 0.38*1080 = 75.6
        // Face should be in upper portion of crop
        let face_cy = 0.45 * 1080.0;
        let face_in_crop = face_cy - crop.y as f64;
        let frac = face_in_crop / crop.h as f64;
        assert!(
            frac > 0.25 && frac < 0.55,
            "face at {frac:.2} from top, expected ~0.38"
        );
    }

    // 2. Face at far right → crop shifts right, clamped
    #[test]
    fn minimal_face_far_right() {
        let faces = [face(85.0, 30.0, 95.0, 60.0, 0.9)];
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        // Face center X = 90% of 1920 = 1728. Crop should be shifted right.
        assert!(crop.x > 600, "crop should be shifted right, got x={}", crop.x);
    }

    // 3. Three faces spread wide → centered, outer faces best-effort
    #[test]
    fn minimal_three_faces_spread() {
        let faces = [
            face(5.0, 30.0, 15.0, 60.0, 0.8),   // left face (10x30 = 300)
            face(40.0, 20.0, 60.0, 60.0, 0.95),  // center face (20x40 = 800, largest = primary)
            face(85.0, 30.0, 95.0, 60.0, 0.7),   // right face (10x30 = 300)
        ];
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        // Primary face (center) should have high visibility
        let vis = face_overlap_fraction(
            &faces[1],
            crop.x as f64,
            crop.y as f64,
            crop.w as f64,
            crop.h as f64,
            1920.0,
            1080.0,
        );
        assert!(vis > 0.7, "primary face visibility {vis:.2} < 0.7");
    }

    // 4. No faces, saliency peak upper-left → crop shifted there
    #[test]
    fn minimal_saliency_upper_left() {
        // Hot spot in the upper-left quadrant of saliency map
        let sal = make_saliency(128, 128, &[
            (10, 10, 1.0),
            (11, 10, 0.9),
            (10, 11, 0.9),
            (12, 10, 0.8),
            (10, 12, 0.8),
        ]);
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &[], Some(&sal), &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        // Crop should be shifted toward the upper-left
        assert!(crop.x < 500, "crop should be left-shifted, got x={}", crop.x);
    }

    // 5. No faces, no saliency → center crop
    #[test]
    fn minimal_no_faces_no_saliency() {
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &[], None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        // Should be roughly centered horizontally
        let center_x = crop.x as f64 + crop.w as f64 / 2.0;
        let diff = (center_x - 960.0).abs();
        assert!(diff < 2.0, "expected centered, center_x={center_x}");
    }

    // 6. Portrait 1080x1920 at 9:16 → full image
    #[test]
    fn minimal_portrait_already_9_16() {
        let config = CropConfig::default();
        let crop = compute_crop(1080, 1920, &[], None, &config).unwrap();

        assert_crop_inside(&crop, 1080, 1920);
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
        assert_eq!(crop.w, 1080);
        assert_eq!(crop.h, 1920);
    }

    // 7. Square 1000x1000 → 562x1000 centered on face
    #[test]
    fn minimal_square_with_face() {
        let faces = [face(40.0, 20.0, 60.0, 50.0, 0.9)];
        let config = CropConfig::default();
        let crop = compute_crop(1000, 1000, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1000, 1000);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.01);
        assert_eq!(crop.h, 1000);
        // Width should be ~562 (1000 * 9/16 = 562.5)
        assert!((crop.w as i32 - 562).abs() <= 1, "expected ~562, got {}", crop.w);
    }

    // 8. Maximal: landscape, large centered face → tight portrait crop with headroom
    #[test]
    fn maximal_landscape_centered_face() {
        // Face occupies a decent portion of the image
        let faces = [face(35.0, 20.0, 65.0, 70.0, 0.95)];
        let config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.02);
        // Should be significantly smaller than the full image
        assert!(
            crop.w < 1920 && crop.h <= 1080,
            "maximal should zoom in: {}x{}",
            crop.w,
            crop.h
        );
    }

    // 9. Maximal: small face in corner → zoomed, clamped
    #[test]
    fn maximal_small_face_corner() {
        let faces = [face(85.0, 75.0, 95.0, 90.0, 0.85)];
        let config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.02);
        // Should be clamped to bottom-right area
        assert!(crop.x + crop.w <= 1920);
        assert!(crop.y + crop.h <= 1080);
    }

    // 10. Maximal: no faces, saliency blob in center → tight around blob
    #[test]
    fn maximal_saliency_center() {
        // Saliency blob in center of 128x128 map
        let sal = make_saliency_rect(128, 128, 50, 50, 78, 78, 1.0);
        let config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let crop = compute_crop(1920, 1080, &[], Some(&sal), &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.02);
        // Should be zoomed in around the center
        assert!(crop.w < 1920, "should zoom in");
    }

    // 11. Maximal: multiple faces close together → includes all
    #[test]
    fn maximal_faces_close() {
        let faces = [
            face(40.0, 30.0, 55.0, 60.0, 0.95), // primary (largest)
            face(55.0, 32.0, 68.0, 58.0, 0.90),  // close neighbor
        ];
        let config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.02);
        // Both faces should be mostly inside the crop
        for f in &faces {
            let vis = face_overlap_fraction(
                f,
                crop.x as f64,
                crop.y as f64,
                crop.w as f64,
                crop.h as f64,
                1920.0,
                1080.0,
            );
            assert!(vis > 0.5, "face visibility {vis:.2} too low");
        }
    }

    // 12. Maximal: multiple faces far apart → zooms on primary
    #[test]
    fn maximal_faces_far_apart() {
        let faces = [
            face(10.0, 30.0, 25.0, 60.0, 0.95), // primary (largest)
            face(80.0, 30.0, 90.0, 55.0, 0.80),  // distant secondary
        ];
        let config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();

        assert_crop_inside(&crop, 1920, 1080);
        assert_approx_aspect(&crop, PORTRAIT_9_16, 0.02);
        // Primary face should be inside the crop
        let vis = face_overlap_fraction(
            &faces[0],
            crop.x as f64,
            crop.y as f64,
            crop.w as f64,
            crop.h as f64,
            1920.0,
            1080.0,
        );
        assert!(vis > 0.7, "primary face visibility {vis:.2} too low");
    }

    // 13. Source matches target aspect → minimal=full image, maximal zooms on subject
    #[test]
    fn source_matches_aspect() {
        let faces = [face(40.0, 30.0, 60.0, 60.0, 0.9)];
        // 9:16 source: 900x1600
        let min_config = CropConfig::default();
        let min_crop = compute_crop(900, 1600, &faces, None, &min_config).unwrap();
        assert_eq!(min_crop.w, 900);
        assert_eq!(min_crop.h, 1600);

        let max_config = CropConfig {
            mode: CropMode::Maximal,
            ..CropConfig::default()
        };
        let max_crop = compute_crop(900, 1600, &faces, None, &max_config).unwrap();
        assert_crop_inside(&max_crop, 900, 1600);
        assert_approx_aspect(&max_crop, PORTRAIT_9_16, 0.02);
        // Maximal should be smaller
        assert!(
            max_crop.w < 900 || max_crop.h < 1600,
            "maximal should zoom in"
        );
    }

    // 14. Degenerate 0-width input → returns None
    #[test]
    fn degenerate_zero_width() {
        let config = CropConfig::default();
        assert!(compute_crop(0, 1080, &[], None, &config).is_none());
    }

    #[test]
    fn degenerate_zero_height() {
        let config = CropConfig::default();
        assert!(compute_crop(1920, 0, &[], None, &config).is_none());
    }

    // Low-confidence faces should be ignored
    #[test]
    fn low_confidence_faces_ignored() {
        let faces = [face(40.0, 30.0, 60.0, 60.0, 0.3)]; // below 0.5
        let config = CropConfig::default();
        let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();
        // Should behave like no-face fallback (centered)
        let center_x = crop.x as f64 + crop.w as f64 / 2.0;
        let diff = (center_x - 960.0).abs();
        assert!(diff < 2.0, "expected centered (no qualifying faces), got center_x={center_x}");
    }

    // All standard aspect ratio constants
    #[test]
    fn all_aspect_ratios_produce_valid_crops() {
        let faces = [face(40.0, 30.0, 60.0, 60.0, 0.9)];
        for &ratio in &[
            PORTRAIT_9_16,
            PORTRAIT_3_4,
            PORTRAIT_4_5,
            SQUARE,
            LANDSCAPE_16_9,
            LANDSCAPE_4_3,
        ] {
            for &mode in &[CropMode::Minimal, CropMode::Maximal] {
                let config = CropConfig {
                    target_aspect: ratio,
                    mode,
                    ..CropConfig::default()
                };
                let crop = compute_crop(1920, 1080, &faces, None, &config).unwrap();
                assert_crop_inside(&crop, 1920, 1080);
                assert_approx_aspect(&crop, ratio, 0.03);
            }
        }
    }
}
