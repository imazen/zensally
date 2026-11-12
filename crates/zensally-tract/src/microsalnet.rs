#![forbid(unsafe_code)]

//! MicroSalNet: lightweight general saliency detection via tract.
//!
//! Custom MobileNetV3-style encoder-decoder (~237K params, ~12ms in tract).
//! Trained on DUTS-TR with knowledge distillation from U2-Netp.
//!
//! Input: NCHW RGB float32 [0, 1], 256x256 (stretched).
//! Output: single-channel saliency map [1, 1, 128, 128] in [0, 1].
//!
//! Uses ConvTranspose2d for upsampling (no Resize ops) to avoid
//! tract's slow bilinear Resize implementation. Output is half
//! input resolution (128x128) — sufficient for smart cropping.

use tract_onnx::prelude::*;
use zensally::{ImageRef, PixelFormat, SaliencyDetector, SaliencyMap};

/// Embedded gzip-compressed MicroSalNet ONNX model.
const MODEL_GZ: &[u8] = include_bytes!("../models/microsalnet.onnx.gz");

const INPUT_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 128;

/// MicroSalNet saliency detector using tract for pure-Rust ONNX inference.
pub struct MicroSalNet {
    model: Arc<TypedRunnableModel>,
    /// Reusable preprocessing buffer: [3, 256, 256] NCHW RGB.
    preprocess_buf: Vec<f32>,
}

impl MicroSalNet {
    /// Create a new MicroSalNet saliency detector.
    pub fn new() -> Result<Self, anyhow::Error> {
        let model_bytes = crate::decompress_gz(MODEL_GZ);
        let proto = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&model_bytes))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
            )?;
        let model = proto.into_optimized()?.into_runnable()?;

        let preprocess_buf = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];

        Ok(Self {
            model,
            preprocess_buf,
        })
    }

    /// Preprocess: bilinear resize to 256x256 (stretch), RGB float32 [0, 1], NCHW layout.
    fn preprocess(
        &mut self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        src_stride: usize,
        format: PixelFormat,
    ) {
        preprocess_256(
            &mut self.preprocess_buf,
            pixels,
            src_w,
            src_h,
            src_stride,
            format,
        );
    }
}

/// Shared 256×256 stretch preprocessing: bilinear resize, RGB float32
/// `[0, 1]`, NCHW layout into `buf` (`3 * 256 * 256` elements).
fn preprocess_256(
    buf: &mut [f32],
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: usize,
    format: PixelFormat,
) {
    let bpp = format.bytes_per_pixel();

    let (r_idx, g_idx, b_idx) = match format {
        PixelFormat::Bgra => (2, 1, 0),
        PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
    };

    let x_ratio = if INPUT_SIZE > 1 {
        (src_w as f32 - 1.0) / (INPUT_SIZE as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if INPUT_SIZE > 1 {
        (src_h as f32 - 1.0) / (INPUT_SIZE as f32 - 1.0)
    } else {
        0.0
    };
    let plane_size = INPUT_SIZE * INPUT_SIZE;

    for dst_y in 0..INPUT_SIZE {
        let src_yf = dst_y as f32 * y_ratio;
        let y0 = src_yf as usize;
        let y1 = (y0 + 1).min(src_h as usize - 1);
        let fy = src_yf - y0 as f32;
        let fy_inv = 1.0 - fy;

        for dst_x in 0..INPUT_SIZE {
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

            let pixel_idx = dst_y * INPUT_SIZE + dst_x;

            // NCHW RGB, scale to [0,1] only (matching training preprocessing)
            buf[pixel_idx] = r / 255.0;
            buf[plane_size + pixel_idx] = g / 255.0;
            buf[2 * plane_size + pixel_idx] = b / 255.0;
        }
    }
}

impl SaliencyDetector for MicroSalNet {
    fn saliency_map(&mut self, image: &ImageRef<'_>) -> SaliencyMap {
        self.preprocess(
            image.pixels,
            image.width,
            image.height,
            image.stride,
            image.format,
        );

        let input = match Tensor::from_shape(
            &[1, 3, INPUT_SIZE, INPUT_SIZE],
            &self.preprocess_buf[..3 * INPUT_SIZE * INPUT_SIZE],
        ) {
            Ok(t) => t,
            Err(_) => {
                return SaliencyMap {
                    data: vec![0.0; OUTPUT_SIZE * OUTPUT_SIZE],
                    width: OUTPUT_SIZE as u32,
                    height: OUTPUT_SIZE as u32,
                };
            }
        };

        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(_) => {
                return SaliencyMap {
                    data: vec![0.0; OUTPUT_SIZE * OUTPUT_SIZE],
                    width: OUTPUT_SIZE as u32,
                    height: OUTPUT_SIZE as u32,
                };
            }
        };

        let raw = match crate::output::plain_f32(&outputs[0]) {
            Ok(s) => s,
            Err(_) => {
                return SaliencyMap {
                    data: vec![0.0; OUTPUT_SIZE * OUTPUT_SIZE],
                    width: OUTPUT_SIZE as u32,
                    height: OUTPUT_SIZE as u32,
                };
            }
        };

        // Output is already sigmoided [0, 1] — just clamp
        let data: Vec<f32> = raw.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        SaliencyMap {
            data,
            width: OUTPUT_SIZE as u32,
            height: OUTPUT_SIZE as u32,
        }
    }
}

/// MicroSalNet v3ds-w24 saliency detector (the v0.2 crop pipeline's model).
///
/// Backed by [`zensally_micro::W24MicroModel`] (hand-scheduled SIMD NHWC
/// inference over `archmage` + `magetypes`, zero heap allocation on warm
/// forward passes) rather than `tract-onnx`.
pub struct MicroSalNetV3 {
    model: zensally_micro::W24MicroModel,
    ws: zensally_micro::W24Workspace,
    /// Reusable preprocessing buffer: `[256, 256, 3]` NHWC RGB in `[0, 1]`.
    nhwc_buf: Vec<f32>,
}

impl MicroSalNetV3 {
    /// Create a new MicroSalNet v3ds-w24 saliency detector.
    pub fn new() -> Result<Self, anyhow::Error> {
        let model = zensally_micro::W24MicroModel::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        let ws = zensally_micro::W24Workspace::new();
        let nhwc_buf = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];

        Ok(Self {
            model,
            ws,
            nhwc_buf,
        })
    }

    /// Downsample a `512×384×3` NHWC RGB `[0, 1]` buffer (from
    /// [`zensally::stream_prep::preprocess_nhwc_strided`]) into `self.nhwc_buf`
    /// (`256×256×3`) and run the zero-allocation [`zensally_micro::W24MicroModel`]
    /// forward pass directly into `out_128x128`.
    #[cfg(all(feature = "yunet", feature = "selfie_seg"))]
    pub(crate) fn forward_from_nhwc_512x384_into(
        &mut self,
        nhwc_512x384: &[f32],
        out_128x128: &mut [f32],
    ) {
        zensally_micro::downsample_512x384_to_256x256_nhwc(nhwc_512x384, &mut self.nhwc_buf);
        self.model
            .forward_nhwc(&self.nhwc_buf, &mut self.ws, out_128x128);
    }

    /// Borrow the most recently prepared `256×256×3` NHWC RGB `[0, 1]` buffer.
    #[cfg(all(feature = "yunet", feature = "selfie_seg"))]
    pub(crate) fn nhwc_256(&self) -> &[f32] {
        &self.nhwc_buf
    }
}

impl SaliencyDetector for MicroSalNetV3 {
    fn saliency_map(&mut self, image: &ImageRef<'_>) -> SaliencyMap {
        zensally_micro::preprocess_image_256x256_nhwc(image, &mut self.nhwc_buf);
        let mut data = vec![0.0f32; OUTPUT_SIZE * OUTPUT_SIZE];
        self.model
            .forward_nhwc(&self.nhwc_buf, &mut self.ws, &mut data);
        SaliencyMap {
            data,
            width: OUTPUT_SIZE as u32,
            height: OUTPUT_SIZE as u32,
        }
    }
}
