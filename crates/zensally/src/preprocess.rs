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

/// Resize an image to `target_w × target_h` and write NCHW RGB float32
/// into `output` (must be at least `3 * target_w * target_h` elements).
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

#[cfg(test)]
mod tests {
    use super::*;

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
            &pixels, 2, 2, PixelFormat::Rgb,
            2, 2, ResizeMode::Stretch, Normalization::UnitScale, &mut output,
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
            &pixels, 2, 2, PixelFormat::Rgb,
            2, 2, ResizeMode::Stretch, Normalization::CenterScale, &mut output,
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
            &pixels, 4, 2, PixelFormat::Rgb,
            4, 4, ResizeMode::Letterbox, Normalization::UnitScale, &mut output,
        );
        assert!((info.ratio - 1.0).abs() < 1e-6, "ratio should be 1.0");
        assert_eq!(info.pad_left, 0.0);
        assert_eq!(info.pad_top, 1.0); // 1 row padding top and bottom
        // Top row (y=0) should be zero padding
        let plane = 4 * 4;
        for x in 0..4 {
            assert!(output[x].abs() < 1e-6, "top pad R[{x}] = {}", output[x]);
        }
        // Middle rows (y=1,2) should have content
        for x in 0..4 {
            let idx = 1 * 4 + x;
            assert!(output[idx] > 0.4, "content R[{x}] = {}", output[idx]);
        }
    }

    #[test]
    fn bgra_channel_reorder() {
        // BGRA pixel: B=10, G=20, R=30, A=255
        let pixels = vec![10, 20, 30, 255, 10, 20, 30, 255,
                          10, 20, 30, 255, 10, 20, 30, 255];
        let mut output = vec![0.0f32; 3 * 2 * 2];
        preprocess_nchw(
            &pixels, 2, 2, PixelFormat::Bgra,
            2, 2, ResizeMode::Stretch, Normalization::UnitScale, &mut output,
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
            &pixels, 1, 1, PixelFormat::Rgb,
            1, 1, ResizeMode::Stretch,
            Normalization::MeanSubtract { r: 50.0, g: 60.0, b: 70.0 },
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
                &pixels, 1, 1, PixelFormat::Rgb,
                1, 1, ResizeMode::Stretch, Normalization::UnitScale, &mut output,
            );
        }));
        assert!(result.is_err(), "should panic on undersized buffer");
    }
}
