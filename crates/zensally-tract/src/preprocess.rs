#![forbid(unsafe_code)]

use zensally::PixelFormat;
use zensally::preprocess::{Normalization, ResizeMode, preprocess_nchw_strided};

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
    let src_stride = src_w as usize * format.bytes_per_pixel();
    preprocess_strided(pixels, src_w, src_h, src_stride, format, target, output)
}

/// Resize a strided source image (`src_stride >= src_w * format.bytes_per_pixel()`)
/// to fit within `target x target` with letterboxing, convert to BGR, subtract mean,
/// and write into `output` in NCHW layout `[3, target, target]`.
pub fn preprocess_strided(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: usize,
    format: PixelFormat,
    target: u32,
    output: &mut [f32],
) -> PreprocessResult {
    let t = target as usize;
    let info = match format {
        PixelFormat::Bgra => preprocess_nchw_strided(
            pixels,
            src_w,
            src_h,
            src_stride,
            PixelFormat::Rgba,
            t,
            t,
            ResizeMode::Letterbox,
            Normalization::MeanSubtract {
                r: MEAN_B,
                g: MEAN_G,
                b: MEAN_R,
            },
            output,
        ),
        PixelFormat::Rgba => preprocess_nchw_strided(
            pixels,
            src_w,
            src_h,
            src_stride,
            PixelFormat::Bgra,
            t,
            t,
            ResizeMode::Letterbox,
            Normalization::MeanSubtract {
                r: MEAN_B,
                g: MEAN_G,
                b: MEAN_R,
            },
            output,
        ),
        PixelFormat::Rgb => {
            let info = preprocess_nchw_strided(
                pixels,
                src_w,
                src_h,
                src_stride,
                PixelFormat::Rgb,
                t,
                t,
                ResizeMode::Letterbox,
                Normalization::MeanSubtract {
                    r: MEAN_R,
                    g: MEAN_G,
                    b: MEAN_B,
                },
                output,
            );
            let plane = t * t;
            let (p0, rest) = output[..3 * plane].split_at_mut(plane);
            let (_, p2) = rest.split_at_mut(plane);
            p0.swap_with_slice(p2);
            info
        }
    };

    PreprocessResult {
        ratio: info.ratio,
        pad_left: info.pad_left as u32,
        pad_top: info.pad_top as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgr_strided_matches_scalar_all_formats() {
        for fmt in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Bgra] {
            let bpp = fmt.bytes_per_pixel();
            let (sw, sh, t) = (31u32, 23u32, 19u32);
            let stride = sw as usize * bpp + 9;
            let mut px = vec![0x55u8; sh as usize * stride];
            for y in 0..sh as usize {
                for x in 0..sw as usize {
                    let o = y * stride + x * bpp;
                    px[o] = ((x * 29 + y * 7) % 256) as u8;
                    px[o + 1] = ((x * 13 + y * 41) % 256) as u8;
                    px[o + 2] = ((x * 3 + y * 59) % 256) as u8;
                    if bpp == 4 {
                        px[o + 3] = 255;
                    }
                }
            }
            let mut got = vec![0.0f32; 3 * t as usize * t as usize];
            let res = preprocess_strided(&px, sw, sh, stride, fmt, t, &mut got);

            let ratio = (t as f32 / sw as f32).min(t as f32 / sh as f32);
            let rw = (sw as f32 * ratio).round() as usize;
            let rh = (sh as f32 * ratio).round() as usize;
            let pl = ((t - rw as u32) / 2) as usize;
            let pt = ((t - rh as u32) / 2) as usize;
            assert_eq!(res.pad_left, pl as u32);
            assert_eq!(res.pad_top, pt as u32);
            let plane = t as usize * t as usize;
            let (ri, gi, bi) = match fmt {
                PixelFormat::Bgra => (2, 1, 0),
                PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
            };
            let xr = (sw as f32 - 1.0) / (rw as f32 - 1.0);
            let yr = (sh as f32 - 1.0) / (rh as f32 - 1.0);
            for dy in 0..rh {
                let sy = dy as f32 * yr;
                let y0 = (sy as usize).min(sh as usize - 1);
                let y1 = (y0 + 1).min(sh as usize - 1);
                let fy = sy - y0 as f32;
                for dx in 0..rw {
                    let sx = dx as f32 * xr;
                    let x0 = (sx as usize).min(sw as usize - 1);
                    let x1 = (x0 + 1).min(sw as usize - 1);
                    let fx = sx - x0 as f32;
                    let o00 = y0 * stride + x0 * bpp;
                    let o10 = y0 * stride + x1 * bpp;
                    let o01 = y1 * stride + x0 * bpp;
                    let o11 = y1 * stride + x1 * bpp;
                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;
                    let r = px[o00 + ri] as f32 * w00
                        + px[o10 + ri] as f32 * w10
                        + px[o01 + ri] as f32 * w01
                        + px[o11 + ri] as f32 * w11;
                    let g = px[o00 + gi] as f32 * w00
                        + px[o10 + gi] as f32 * w10
                        + px[o01 + gi] as f32 * w01
                        + px[o11 + gi] as f32 * w11;
                    let b = px[o00 + bi] as f32 * w00
                        + px[o10 + bi] as f32 * w10
                        + px[o01 + bi] as f32 * w01
                        + px[o11 + bi] as f32 * w11;
                    let idx = (dy + pt) * t as usize + dx + pl;
                    assert!((got[idx] - (b - MEAN_B)).abs() < 1e-4);
                    assert!((got[plane + idx] - (g - MEAN_G)).abs() < 1e-4);
                    assert!((got[2 * plane + idx] - (r - MEAN_R)).abs() < 1e-4);
                }
            }
        }
    }
}
