//! Shared image preprocessing for ONNX face/saliency models.
//!
//! Bilinear resize to NCHW float32 layout with configurable normalization.
//! Supports letterboxing (aspect-preserving pad) or stretch modes.

use crate::PixelFormat;
use archmage::prelude::*;
use magetypes::simd::backends::F32x8Convert as Backend;
use magetypes::simd::generic::{f32x8 as V, i32x8 as I};

/// How to handle aspect ratio mismatch during resize.
#[derive(Debug, Clone, Copy)]
pub enum ResizeMode {
    /// Aspect-preserving resize, centered with padding.
    /// Returns the letterbox metadata for coordinate reversal.
    Letterbox,
    /// Stretch to fill the target dimensions.
    Stretch,
}

/// How to normalize pixel values after resize.
#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    /// (pixel - 127) / 128 → approximately [-1, 1]. Used by UltraFace.
    CenterScale,
    /// pixel / 255 → [0, 1]. Used by MicroSalNet, Selfie Segmentation.
    UnitScale,
    /// pixel - [mean_r, mean_g, mean_b]. Used by BlazeFace.
    MeanSubtract { r: f32, g: f32, b: f32 },
}

/// Result of letterbox preprocessing — needed to reverse coordinate transforms.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    /// Scale ratio applied (original → resized).
    pub ratio: f32,
    /// X offset of content within padded tensor.
    pub pad_left: f32,
    /// Y offset of content within padded tensor.
    pub pad_top: f32,
}

/// Preprocessing configuration for ONNX model input.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    /// Target tensor width.
    pub target_w: usize,
    /// Target tensor height.
    pub target_h: usize,
    /// How to handle aspect ratio mismatch.
    pub mode: ResizeMode,
    /// Pixel value normalization.
    pub norm: Normalization,
}

/// Resize a tightly-packed image to `target_w × target_h` and write NCHW RGB
/// float32 into `output` (must be at least `3 * target_w * target_h` elements).
///
/// Returns [`LetterboxInfo`] if using [`ResizeMode::Letterbox`], or a
/// trivial info (ratio=1, pad=0) for stretch mode.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_nchw(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    format: PixelFormat,
    target_w: usize,
    target_h: usize,
    mode: ResizeMode,
    norm: Normalization,
    output: &mut [f32],
) -> LetterboxInfo {
    let src_stride = src_w as usize * format.bytes_per_pixel();
    preprocess_nchw_strided(
        pixels, src_w, src_h, src_stride, format, target_w, target_h, mode, norm, output,
    )
}

/// Resize a strided image (`src_stride >= src_w * format.bytes_per_pixel()`)
/// to `target_w × target_h` and write NCHW RGB float32 into `output`
/// (must be at least `3 * target_w * target_h` elements).
///
/// Returns [`LetterboxInfo`] if using [`ResizeMode::Letterbox`], or a
/// trivial info (ratio=1, pad=0) for stretch mode.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_nchw_strided(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: usize,
    format: PixelFormat,
    target_w: usize,
    target_h: usize,
    mode: ResizeMode,
    norm: Normalization,
    output: &mut [f32],
) -> LetterboxInfo {
    let bpp = format.bytes_per_pixel();
    let packed = src_w as usize * bpp;
    assert!(src_stride >= packed, "src_stride {src_stride} < {packed}");
    let total = 3 * target_w * target_h;
    assert!(
        output.len() >= total,
        "output buffer too small: need {total} got {}",
        output.len()
    );

    let (resized_w, resized_h, pad_left, pad_top, ratio) = match mode {
        ResizeMode::Letterbox => {
            let ratio = (target_w as f32 / src_w as f32).min(target_h as f32 / src_h as f32);
            let rw = (src_w as f32 * ratio).round() as usize;
            let rh = (src_h as f32 * ratio).round() as usize;
            let pl = (target_w - rw) / 2;
            let pt = (target_h - rh) / 2;
            (rw, rh, pl, pt, ratio)
        }
        ResizeMode::Stretch => (target_w, target_h, 0, 0, 1.0),
    };

    // Fill with neutral padding value (depends on normalization).
    let pad_val = match norm {
        Normalization::CenterScale => 0.0,         // (127-127)/128
        Normalization::UnitScale => 0.0,           // black
        Normalization::MeanSubtract { .. } => 0.0, // mean-subtracted zero
    };
    output[..total].fill(pad_val);

    let plane_size = target_w * target_h;
    if resized_w == 0 || resized_h == 0 || src_w == 0 || src_h == 0 {
        return LetterboxInfo {
            ratio,
            pad_left: pad_left as f32,
            pad_top: pad_top as f32,
        };
    }

    let max_x = src_w as usize - 1;
    let max_y = src_h as usize - 1;

    let x_ratio = if resized_w > 1 {
        max_x as f32 / (resized_w as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if resized_h > 1 {
        max_y as f32 / (resized_h as f32 - 1.0)
    } else {
        0.0
    };

    let is_bgra = matches!(format, PixelFormat::Bgra);
    let (scale_r, off_r, scale_g, off_g, scale_b, off_b) = match norm {
        Normalization::CenterScale => (
            1.0 / 128.0,
            -127.0 / 128.0,
            1.0 / 128.0,
            -127.0 / 128.0,
            1.0 / 128.0,
            -127.0 / 128.0,
        ),
        Normalization::UnitScale => (1.0 / 255.0, 0.0, 1.0 / 255.0, 0.0, 1.0 / 255.0, 0.0),
        Normalization::MeanSubtract { r, g, b } => (1.0, -r, 1.0, -g, 1.0, -b),
    };

    let (r_plane, rest) = output[..total].split_at_mut(plane_size);
    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

    for dst_y in 0..resized_h {
        let src_yf = dst_y as f32 * y_ratio;
        let y0 = (src_yf as usize).min(max_y);
        let y1 = (y0 + 1).min(max_y);
        let fy = src_yf - y0 as f32;
        let out_off = (dst_y + pad_top) * target_w + pad_left;

        let row0 = &pixels[y0 * src_stride..y0 * src_stride + packed];
        let row1 = &pixels[y1 * src_stride..y1 * src_stride + packed];
        let r_out = &mut r_plane[out_off..out_off + resized_w];
        let g_out = &mut g_plane[out_off..out_off + resized_w];
        let b_out = &mut b_plane[out_off..out_off + resized_w];

        attuned!(preprocess_nchw_row(
            row0, row1, bpp, is_bgra, max_x, resized_w, x_ratio, fy, scale_r, off_r, scale_g,
            off_g, scale_b, off_b, r_out, g_out, b_out
        ));
    }

    LetterboxInfo {
        ratio,
        pad_left: pad_left as f32,
        pad_top: pad_top as f32,
    }
}

#[inline(always)]
fn load_rgb3_packed(row: &[u8], o: usize) -> i32 {
    if o + 4 <= row.len() {
        i32::from_le_bytes(row[o..o + 4].try_into().unwrap())
    } else {
        (row[o] as i32) | ((row[o + 1] as i32) << 8) | ((row[o + 2] as i32) << 16)
    }
}

#[inline(always)]
fn unpack_rgb_f32<T: Backend>(packed: I<T>, mask_ff: I<T>) -> (V<T>, V<T>, V<T>) {
    let c0 = (packed & mask_ff).to_f32();
    let c1 = (packed.shr_logical_const::<8>() & mask_ff).to_f32();
    let c2 = (packed.shr_logical_const::<16>() & mask_ff).to_f32();
    (c0, c1, c2)
}

#[attune(_*_t)]
#[allow(clippy::too_many_arguments)]
fn preprocess_nchw_row(
    token: Token,
    row0: &[u8],
    row1: &[u8],
    bpp: usize,
    is_bgra: bool,
    max_x: usize,
    rw: usize,
    x_ratio: f32,
    fy: f32,
    scale_r: f32,
    off_r: f32,
    scale_g: f32,
    off_g: f32,
    scale_b: f32,
    off_b: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
) {
    preprocess_nchw_row_inner(
        token, row0, row1, bpp, is_bgra, max_x, rw, x_ratio, fy, scale_r, off_r, scale_g, off_g,
        scale_b, off_b, r_out, g_out, b_out,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn preprocess_nchw_row_inner<T: Backend>(
    token: T,
    row0: &[u8],
    row1: &[u8],
    bpp: usize,
    is_bgra: bool,
    max_x: usize,
    rw: usize,
    x_ratio: f32,
    fy: f32,
    scale_r: f32,
    off_r: f32,
    scale_g: f32,
    off_g: f32,
    scale_b: f32,
    off_b: f32,
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
) {
    let one = V::<T>::splat_t(token, 1.0);
    let zero_v = V::<T>::zero_t(token);
    let lanes = V::<T>::from_array_t(token, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let x_ratio_v = V::<T>::splat_t(token, x_ratio);
    let fy_v = V::<T>::splat_t(token, fy);
    let fy_inv_v = one - fy_v;
    let max_x_i32 = I::<T>::splat_t(token, max_x as i32);
    let mask_ff = I::<T>::splat_t(token, 0xff);

    let sr_v = V::<T>::splat_t(token, scale_r);
    let or_v = V::<T>::splat_t(token, off_r);
    let sg_v = V::<T>::splat_t(token, scale_g);
    let og_v = V::<T>::splat_t(token, off_g);
    let sb_v = V::<T>::splat_t(token, scale_b);
    let ob_v = V::<T>::splat_t(token, off_b);

    let mut b8 = 0usize;
    while b8 + 8 <= rw {
        let xfv = ((V::<T>::splat_t(token, b8 as f32) + lanes) * x_ratio_v).max_fast(zero_v);
        let x0_i32 = xfv.to_i32_fast();
        let fx = xfv - x0_i32.to_f32();
        let fx_inv = one - fx;

        let x0_clamped = x0_i32.min(max_x_i32);
        let x0_a = x0_clamped.to_array();
        let x1_a = (x0_clamped + I::<T>::splat_t(token, 1))
            .min(max_x_i32)
            .to_array();

        let w00 = fx_inv * fy_inv_v;
        let w10 = fx * fy_inv_v;
        let w01 = fx_inv * fy_v;
        let w11 = fx * fy_v;

        let mut p00 = [0i32; 8];
        let mut p10 = [0i32; 8];
        let mut p01 = [0i32; 8];
        let mut p11 = [0i32; 8];

        if bpp == 4 {
            for k in 0..8 {
                let o0 = x0_a[k] as usize * 4;
                let o1 = x1_a[k] as usize * 4;
                p00[k] = i32::from_le_bytes(row0[o0..o0 + 4].try_into().unwrap());
                p10[k] = i32::from_le_bytes(row0[o1..o1 + 4].try_into().unwrap());
                p01[k] = i32::from_le_bytes(row1[o0..o0 + 4].try_into().unwrap());
                p11[k] = i32::from_le_bytes(row1[o1..o1 + 4].try_into().unwrap());
            }
        } else {
            for k in 0..8 {
                let o0 = x0_a[k] as usize * bpp;
                let o1 = x1_a[k] as usize * bpp;
                p00[k] = load_rgb3_packed(row0, o0);
                p10[k] = load_rgb3_packed(row0, o1);
                p01[k] = load_rgb3_packed(row1, o0);
                p11[k] = load_rgb3_packed(row1, o1);
            }
        }

        let (c0_00, c1_00, c2_00) = unpack_rgb_f32(I::<T>::from_array_t(token, p00), mask_ff);
        let (c0_10, c1_10, c2_10) = unpack_rgb_f32(I::<T>::from_array_t(token, p10), mask_ff);
        let (c0_01, c1_01, c2_01) = unpack_rgb_f32(I::<T>::from_array_t(token, p01), mask_ff);
        let (c0_11, c1_11, c2_11) = unpack_rgb_f32(I::<T>::from_array_t(token, p11), mask_ff);

        let c0 = c0_00.mul_add_relaxed(
            w00,
            c0_10.mul_add_relaxed(w10, c0_01.mul_add_relaxed(w01, c0_11 * w11)),
        );
        let c1 = c1_00.mul_add_relaxed(
            w00,
            c1_10.mul_add_relaxed(w10, c1_01.mul_add_relaxed(w01, c1_11 * w11)),
        );
        let c2 = c2_00.mul_add_relaxed(
            w00,
            c2_10.mul_add_relaxed(w10, c2_01.mul_add_relaxed(w01, c2_11 * w11)),
        );

        let (r, g, b) = if is_bgra { (c2, c1, c0) } else { (c0, c1, c2) };
        r.mul_add_relaxed(sr_v, or_v)
            .store((&mut r_out[b8..b8 + 8]).try_into().unwrap());
        g.mul_add_relaxed(sg_v, og_v)
            .store((&mut g_out[b8..b8 + 8]).try_into().unwrap());
        b.mul_add_relaxed(sb_v, ob_v)
            .store((&mut b_out[b8..b8 + 8]).try_into().unwrap());

        b8 += 8;
    }

    if b8 < rw {
        let (r_idx, g_idx, b_idx) = if is_bgra { (2, 1, 0) } else { (0, 1, 2) };
        let fy_inv = 1.0 - fy;
        for dst_x in b8..rw {
            let src_xf = dst_x as f32 * x_ratio;
            let x0 = (src_xf as usize).min(max_x);
            let x1 = (x0 + 1).min(max_x);
            let fx = src_xf - x0 as f32;
            let fx_inv = 1.0 - fx;

            let o0 = x0 * bpp;
            let o1 = x1 * bpp;

            let w00 = fx_inv * fy_inv;
            let w10 = fx * fy_inv;
            let w01 = fx_inv * fy;
            let w11 = fx * fy;

            let r = row0[o0 + r_idx] as f32 * w00
                + row0[o1 + r_idx] as f32 * w10
                + row1[o0 + r_idx] as f32 * w01
                + row1[o1 + r_idx] as f32 * w11;
            let g = row0[o0 + g_idx] as f32 * w00
                + row0[o1 + g_idx] as f32 * w10
                + row1[o0 + g_idx] as f32 * w01
                + row1[o1 + g_idx] as f32 * w11;
            let b = row0[o0 + b_idx] as f32 * w00
                + row0[o1 + b_idx] as f32 * w10
                + row1[o0 + b_idx] as f32 * w01
                + row1[o1 + b_idx] as f32 * w11;

            r_out[dst_x] = r.mul_add(scale_r, off_r);
            g_out[dst_x] = g.mul_add(scale_g, off_g);
            b_out[dst_x] = b.mul_add(scale_b, off_b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn nchw_simd_matches_scalar_across_formats_strides_and_tails() {
        #[allow(clippy::too_many_arguments)]
        fn scalar_ref(
            pixels: &[u8],
            src_w: u32,
            src_h: u32,
            src_stride: usize,
            format: PixelFormat,
            tw: usize,
            th: usize,
            mode: ResizeMode,
            norm: Normalization,
        ) -> alloc::vec::Vec<f32> {
            let bpp = format.bytes_per_pixel();
            let (r_idx, g_idx, b_idx) = match format {
                PixelFormat::Bgra => (2, 1, 0),
                PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
            };
            let (rw, rh, pl, pt) = match mode {
                ResizeMode::Letterbox => {
                    let ratio = (tw as f32 / src_w as f32).min(th as f32 / src_h as f32);
                    let rw = (src_w as f32 * ratio).round() as usize;
                    let rh = (src_h as f32 * ratio).round() as usize;
                    (rw, rh, (tw - rw) / 2, (th - rh) / 2)
                }
                ResizeMode::Stretch => (tw, th, 0, 0),
            };
            let plane = tw * th;
            let mut out = alloc::vec![0.0f32; 3 * plane];
            let xr = if rw > 1 {
                (src_w as f32 - 1.0) / (rw as f32 - 1.0)
            } else {
                0.0
            };
            let yr = if rh > 1 {
                (src_h as f32 - 1.0) / (rh as f32 - 1.0)
            } else {
                0.0
            };
            for dy in 0..rh {
                let syf = dy as f32 * yr;
                let y0 = (syf as usize).min(src_h as usize - 1);
                let y1 = (y0 + 1).min(src_h as usize - 1);
                let fy = syf - y0 as f32;
                let fy_inv = 1.0 - fy;
                for dx in 0..rw {
                    let sxf = dx as f32 * xr;
                    let x0 = (sxf as usize).min(src_w as usize - 1);
                    let x1 = (x0 + 1).min(src_w as usize - 1);
                    let fx = sxf - x0 as f32;
                    let fx_inv = 1.0 - fx;
                    let w00 = fx_inv * fy_inv;
                    let w10 = fx * fy_inv;
                    let w01 = fx_inv * fy;
                    let w11 = fx * fy;
                    let o00 = y0 * src_stride + x0 * bpp;
                    let o10 = y0 * src_stride + x1 * bpp;
                    let o01 = y1 * src_stride + x0 * bpp;
                    let o11 = y1 * src_stride + x1 * bpp;
                    let r = pixels[o00 + r_idx] as f32 * w00
                        + pixels[o10 + r_idx] as f32 * w10
                        + pixels[o01 + r_idx] as f32 * w01
                        + pixels[o11 + r_idx] as f32 * w11;
                    let g = pixels[o00 + g_idx] as f32 * w00
                        + pixels[o10 + g_idx] as f32 * w10
                        + pixels[o01 + g_idx] as f32 * w01
                        + pixels[o11 + g_idx] as f32 * w11;
                    let b = pixels[o00 + b_idx] as f32 * w00
                        + pixels[o10 + b_idx] as f32 * w10
                        + pixels[o01 + b_idx] as f32 * w01
                        + pixels[o11 + b_idx] as f32 * w11;
                    let (nr, ng, nb) = match norm {
                        Normalization::CenterScale => (
                            (r - 127.0) / 128.0,
                            (g - 127.0) / 128.0,
                            (b - 127.0) / 128.0,
                        ),
                        Normalization::UnitScale => (r / 255.0, g / 255.0, b / 255.0),
                        Normalization::MeanSubtract {
                            r: mr,
                            g: mg,
                            b: mb,
                        } => (r - mr, g - mg, b - mb),
                    };
                    let idx = (dy + pt) * tw + dx + pl;
                    out[idx] = nr;
                    out[plane + idx] = ng;
                    out[2 * plane + idx] = nb;
                }
            }
            out
        }

        for fmt in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Bgra] {
            let bpp = fmt.bytes_per_pixel();
            let (sw, sh) = (37u32, 29u32);
            let stride = sw as usize * bpp + 13;
            let mut pixels = alloc::vec![0xAAu8; sh as usize * stride];
            for y in 0..sh as usize {
                for x in 0..sw as usize {
                    let o = y * stride + x * bpp;
                    pixels[o] = ((x * 17 + y * 31) % 256) as u8;
                    pixels[o + 1] = ((x * 53 + y * 11) % 256) as u8;
                    pixels[o + 2] = ((x * 7 + y * 43) % 256) as u8;
                    if bpp == 4 {
                        pixels[o + 3] = 255;
                    }
                }
            }
            for norm in [
                Normalization::CenterScale,
                Normalization::UnitScale,
                Normalization::MeanSubtract {
                    r: 103.5,
                    g: 116.2,
                    b: 123.7,
                },
            ] {
                for (tw, th, mode) in [
                    (23, 19, ResizeMode::Stretch),
                    (25, 25, ResizeMode::Letterbox),
                    (64, 48, ResizeMode::Letterbox),
                ] {
                    let mut simd = alloc::vec![0.0f32; 3 * tw * th];
                    preprocess_nchw_strided(
                        &pixels, sw, sh, stride, fmt, tw, th, mode, norm, &mut simd,
                    );
                    let ref_out = scalar_ref(&pixels, sw, sh, stride, fmt, tw, th, mode, norm);
                    for (i, (&a, &b)) in simd.iter().zip(ref_out.iter()).enumerate() {
                        assert!(
                            (a - b).abs() < 1e-4,
                            "mismatch at {i}: {a} vs {b} ({fmt:?} {norm:?})"
                        );
                    }
                }
            }
        }
    }

    /// 2x2 red image in RGB format.
    fn red_2x2_rgb() -> Vec<u8> {
        vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]
    }

    #[test]
    fn stretch_identity_unit_scale() {
        // 2x2 → 2x2 stretch with UnitScale should give R=1.0, G=B=0.0
        let pixels = red_2x2_rgb();
        let mut output = vec![0.0f32; 3 * 2 * 2];
        let info = preprocess_nchw(
            &pixels,
            2,
            2,
            PixelFormat::Rgb,
            2,
            2,
            ResizeMode::Stretch,
            Normalization::UnitScale,
            &mut output,
        );
        assert!((info.ratio - 1.0).abs() < 1e-6);
        assert_eq!(info.pad_left, 0.0);
        assert_eq!(info.pad_top, 0.0);
        // NCHW: plane 0 = R, plane 1 = G, plane 2 = B
        let plane = 2 * 2;
        for i in 0..4 {
            assert!((output[i] - 1.0).abs() < 1e-4, "R[{i}] = {}", output[i]);
            assert!(output[plane + i].abs() < 1e-4, "G[{i}]");
            assert!(output[2 * plane + i].abs() < 1e-4, "B[{i}]");
        }
    }

    #[test]
    fn center_scale_normalization() {
        // Pure white (255,255,255) → (255-127)/128 = 1.0
        let pixels = vec![255u8; 3 * 2 * 2];
        let mut output = vec![0.0f32; 3 * 2 * 2];
        preprocess_nchw(
            &pixels,
            2,
            2,
            PixelFormat::Rgb,
            2,
            2,
            ResizeMode::Stretch,
            Normalization::CenterScale,
            &mut output,
        );
        for &v in &output {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn letterbox_landscape_into_square() {
        // 4x2 → 4x4 letterbox: image fills width, padded top/bottom
        let pixels = vec![128u8; 3 * 4 * 2];
        let mut output = vec![-99.0f32; 3 * 4 * 4];
        let info = preprocess_nchw(
            &pixels,
            4,
            2,
            PixelFormat::Rgb,
            4,
            4,
            ResizeMode::Letterbox,
            Normalization::UnitScale,
            &mut output,
        );
        assert!((info.ratio - 1.0).abs() < 1e-6, "ratio should be 1.0");
        assert_eq!(info.pad_left, 0.0);
        assert_eq!(info.pad_top, 1.0); // 1 row padding top and bottom
        // Top row (y=0) should be zero padding
        for (x, val) in output[..4].iter().enumerate() {
            assert!(val.abs() < 1e-6, "top pad R[{x}] = {val}");
        }
        // Middle rows (y=1,2) should have content
        for (x, val) in output[4..8].iter().enumerate() {
            assert!(*val > 0.4, "content R[{x}] = {val}");
        }
    }

    #[test]
    fn bgra_channel_reorder() {
        // BGRA pixel: B=10, G=20, R=30, A=255
        let pixels = vec![
            10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255,
        ];
        let mut output = vec![0.0f32; 3 * 2 * 2];
        preprocess_nchw(
            &pixels,
            2,
            2,
            PixelFormat::Bgra,
            2,
            2,
            ResizeMode::Stretch,
            Normalization::UnitScale,
            &mut output,
        );
        let plane = 4;
        // NCHW RGB: plane 0 = R (30/255), plane 1 = G (20/255), plane 2 = B (10/255)
        assert!((output[0] - 30.0 / 255.0).abs() < 1e-4, "R");
        assert!((output[plane] - 20.0 / 255.0).abs() < 1e-4, "G");
        assert!((output[2 * plane] - 10.0 / 255.0).abs() < 1e-4, "B");
    }

    #[test]
    fn mean_subtract_normalization() {
        let pixels = vec![100u8; 3]; // 1x1 RGB (100, 100, 100)
        let mut output = vec![0.0f32; 3];
        preprocess_nchw(
            &pixels,
            1,
            1,
            PixelFormat::Rgb,
            1,
            1,
            ResizeMode::Stretch,
            Normalization::MeanSubtract {
                r: 50.0,
                g: 60.0,
                b: 70.0,
            },
            &mut output,
        );
        assert!((output[0] - 50.0).abs() < 1e-4, "R: 100-50=50");
        assert!((output[1] - 40.0).abs() < 1e-4, "G: 100-60=40");
        assert!((output[2] - 30.0).abs() < 1e-4, "B: 100-70=30");
    }

    #[test]
    fn output_buffer_size_check() {
        let pixels = vec![0u8; 3];
        let mut output = vec![0.0f32; 2]; // too small for 1x1x3
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            preprocess_nchw(
                &pixels,
                1,
                1,
                PixelFormat::Rgb,
                1,
                1,
                ResizeMode::Stretch,
                Normalization::UnitScale,
                &mut output,
            );
        }));
        assert!(result.is_err(), "should panic on undersized buffer");
    }
}
