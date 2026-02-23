#![forbid(unsafe_code)]

use zenfaces::PixelFormat;

/// BGR mean values for normalization (standard face detection means).
const MEAN_B: f32 = 104.0;
const MEAN_G: f32 = 117.0;
const MEAN_R: f32 = 123.0;

/// Result of preprocessing, needed to map detections back to original coordinates.
#[derive(Debug)]
pub struct PreprocessResult {
    /// Scale ratio applied during resize (original → resized).
    pub ratio: f32,
    /// X offset of the resized image within the padded tensor.
    pub pad_left: u32,
    /// Y offset of the resized image within the padded tensor.
    pub pad_top: u32,
}

/// Resize source image to fit within `target x target` with letterboxing, convert to BGR,
/// subtract mean, and write into `output` in NCHW layout `[3, target, target]`.
///
/// The border is filled with the mean color (which becomes 0.0 after subtraction).
pub fn preprocess(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    format: PixelFormat,
    target: u32,
    output: &mut [f32],
) -> PreprocessResult {
    let bpp = format.bytes_per_pixel();
    let t = target as usize;

    assert!(
        output.len() >= 3 * t * t,
        "output buffer too small: need {} got {}",
        3 * t * t,
        output.len()
    );

    // Aspect-preserving scale to fit within target x target
    let ratio = (target as f32 / src_w as f32).min(target as f32 / src_h as f32);
    let resized_w = (src_w as f32 * ratio).round() as u32;
    let resized_h = (src_h as f32 * ratio).round() as u32;

    // Center the resized image in the target square
    let pad_left = (target - resized_w) / 2;
    let pad_top = (target - resized_h) / 2;

    // Fill with zeros (mean-subtracted border color)
    output[..3 * t * t].fill(0.0);

    let plane_size = t * t;

    // Extract RGB indices based on pixel format
    let (r_idx, g_idx, b_idx) = match format {
        PixelFormat::Bgra => (2, 1, 0),
        PixelFormat::Rgba => (0, 1, 2),
        PixelFormat::Rgb => (0, 1, 2),
    };

    // Bilinear interpolation ratios
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

    let src_stride = src_w as usize * bpp;

    for dst_y in 0..resized_h as usize {
        let src_yf = dst_y as f32 * y_ratio;
        let y0 = src_yf as usize;
        let y1 = (y0 + 1).min(src_h as usize - 1);
        let fy = src_yf - y0 as f32;
        let fy_inv = 1.0 - fy;

        let out_y = dst_y + pad_top as usize;

        for dst_x in 0..resized_w as usize {
            let src_xf = dst_x as f32 * x_ratio;
            let x0 = src_xf as usize;
            let x1 = (x0 + 1).min(src_w as usize - 1);
            let fx = src_xf - x0 as f32;
            let fx_inv = 1.0 - fx;

            // Four source pixel offsets
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

            let out_x = dst_x + pad_left as usize;
            let pixel_idx = out_y * t + out_x;

            // NCHW BGR order with mean subtraction
            output[pixel_idx] = b - MEAN_B;
            output[plane_size + pixel_idx] = g - MEAN_G;
            output[2 * plane_size + pixel_idx] = r - MEAN_R;
        }
    }

    PreprocessResult {
        ratio,
        pad_left,
        pad_top,
    }
}
