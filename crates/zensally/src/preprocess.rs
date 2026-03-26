//! Shared image preprocessing for ONNX face/saliency models.
//!
//! Bilinear resize to NCHW float32 layout with configurable normalization.
//! Supports letterboxing (aspect-preserving pad) or stretch modes.

use crate::PixelFormat;

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

/// Resize an image to `target_w × target_h` and write NCHW RGB float32
/// into `output` (must be at least `3 * target_w * target_h` elements).
///
/// Returns [`LetterboxInfo`] if using [`ResizeMode::Letterbox`], or a
/// trivial info (ratio=1, pad=0) for stretch mode.
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
    let bpp = format.bytes_per_pixel();
    let total = 3 * target_w * target_h;
    assert!(
        output.len() >= total,
        "output buffer too small: need {total} got {}",
        output.len()
    );

    let (r_idx, g_idx, b_idx) = match format {
        PixelFormat::Bgra => (2, 1, 0),
        PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
    };

    let (resized_w, resized_h, pad_left, pad_top, ratio) = match mode {
        ResizeMode::Letterbox => {
            let ratio =
                (target_w as f32 / src_w as f32).min(target_h as f32 / src_h as f32);
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
        Normalization::CenterScale => 0.0,      // (127-127)/128
        Normalization::UnitScale => 0.0,         // black
        Normalization::MeanSubtract { .. } => 0.0, // mean-subtracted zero
    };
    output[..total].fill(pad_val);

    let plane_size = target_w * target_h;
    let src_stride = src_w as usize * bpp;

    let x_ratio = if resized_w > 1 {
        (src_w as f32 - 1.0) / (resized_w as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if resized_h > 1 {
        (src_h as f32 - 1.0) / (resized_h as f32 - 1.0)
    } else {
        0.0
    };

    for dst_y in 0..resized_h {
        let src_yf = dst_y as f32 * y_ratio;
        let y0 = src_yf as usize;
        let y1 = (y0 + 1).min(src_h as usize - 1);
        let fy = src_yf - y0 as f32;
        let fy_inv = 1.0 - fy;
        let out_y = dst_y + pad_top;

        for dst_x in 0..resized_w {
            let src_xf = dst_x as f32 * x_ratio;
            let x0 = src_xf as usize;
            let x1 = (x0 + 1).min(src_w as usize - 1);
            let fx = src_xf - x0 as f32;
            let fx_inv = 1.0 - fx;

            let off00 = y0 * src_stride + x0 * bpp;
            let off10 = y0 * src_stride + x1 * bpp;
            let off01 = y1 * src_stride + x0 * bpp;
            let off11 = y1 * src_stride + x1 * bpp;

            let w00 = fx_inv * fy_inv;
            let w10 = fx * fy_inv;
            let w01 = fx_inv * fy;
            let w11 = fx * fy;

            let r = pixels[off00 + r_idx] as f32 * w00
                + pixels[off10 + r_idx] as f32 * w10
                + pixels[off01 + r_idx] as f32 * w01
                + pixels[off11 + r_idx] as f32 * w11;
            let g = pixels[off00 + g_idx] as f32 * w00
                + pixels[off10 + g_idx] as f32 * w10
                + pixels[off01 + g_idx] as f32 * w01
                + pixels[off11 + g_idx] as f32 * w11;
            let b = pixels[off00 + b_idx] as f32 * w00
                + pixels[off10 + b_idx] as f32 * w10
                + pixels[off01 + b_idx] as f32 * w01
                + pixels[off11 + b_idx] as f32 * w11;

            let (nr, ng, nb) = match norm {
                Normalization::CenterScale => {
                    ((r - 127.0) / 128.0, (g - 127.0) / 128.0, (b - 127.0) / 128.0)
                }
                Normalization::UnitScale => (r / 255.0, g / 255.0, b / 255.0),
                Normalization::MeanSubtract {
                    r: mr,
                    g: mg,
                    b: mb,
                } => (r - mr, g - mg, b - mb),
            };

            let out_x = dst_x + pad_left;
            let pixel_idx = out_y * target_w + out_x;

            // NCHW RGB layout
            output[pixel_idx] = nr;
            output[plane_size + pixel_idx] = ng;
            output[2 * plane_size + pixel_idx] = nb;
        }
    }

    LetterboxInfo {
        ratio,
        pad_left: pad_left as f32,
        pad_top: pad_top as f32,
    }
}
