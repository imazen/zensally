#![forbid(unsafe_code)]

//! U2-Netp salient object detection via tract.
//!
//! Small variant of U-squared Net (1.13M params, ~4.5MB ONNX).
//! Input: NCHW RGB float32, ImageNet-normalized.
//! Output: single-channel saliency map [1, 1, 320, 320] in [0, 1].
//!
//! The ONNX model is pre-simplified (onnxsim) and patched for tract
//! compatibility: coordinate_transformation_mode pytorch_half_pixel →
//! half_pixel, and sizes-based Resize converted to scales-based.

use tract_onnx::prelude::*;
use zensally::{ImageRef, PixelFormat, SaliencyDetector, SaliencyMap};

/// Embedded gzip-compressed U2-Netp ONNX model (single-output, simplified, patched).
const MODEL_GZ: &[u8] = include_bytes!("../models/u2netp.onnx.gz");

const INPUT_SIZE: usize = 320;

/// ImageNet normalization constants.
const MEAN_R: f32 = 0.485;
const MEAN_G: f32 = 0.456;
const MEAN_B: f32 = 0.406;
const STD_R: f32 = 0.229;
const STD_G: f32 = 0.224;
const STD_B: f32 = 0.225;

/// U2-Netp salient object detector using tract for pure-Rust ONNX inference.
pub struct U2NetpDetector {
    model: TypedRunnableModel<TypedModel>,
    /// Reusable preprocessing buffer: [3, 320, 320] NCHW RGB.
    preprocess_buf: Vec<f32>,
}

impl U2NetpDetector {
    /// Create a new saliency detector.
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

    /// Preprocess: bilinear resize to 320x320 (stretch, no letterbox),
    /// RGB float32, ImageNet-normalized, NCHW layout.
    fn preprocess(&mut self, pixels: &[u8], src_w: u32, src_h: u32, format: PixelFormat) {
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
        let src_stride = src_w as usize * bpp;
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

                // NCHW RGB, scale to [0,1] then ImageNet normalize
                self.preprocess_buf[pixel_idx] = (r / 255.0 - MEAN_R) / STD_R;
                self.preprocess_buf[plane_size + pixel_idx] = (g / 255.0 - MEAN_G) / STD_G;
                self.preprocess_buf[2 * plane_size + pixel_idx] = (b / 255.0 - MEAN_B) / STD_B;
            }
        }
    }
}

impl SaliencyDetector for U2NetpDetector {
    fn saliency_map(&mut self, image: &ImageRef<'_>) -> SaliencyMap {
        self.preprocess(image.pixels, image.width, image.height, image.format);

        let input = match Tensor::from_shape(
            &[1, 3, INPUT_SIZE, INPUT_SIZE],
            &self.preprocess_buf[..3 * INPUT_SIZE * INPUT_SIZE],
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("U2Netp tensor creation failed: {e}");
                return SaliencyMap {
                    data: vec![0.0; INPUT_SIZE * INPUT_SIZE],
                    width: INPUT_SIZE as u32,
                    height: INPUT_SIZE as u32,
                };
            }
        };

        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("U2Netp inference failed: {e}");
                return SaliencyMap {
                    data: vec![0.0; INPUT_SIZE * INPUT_SIZE],
                    width: INPUT_SIZE as u32,
                    height: INPUT_SIZE as u32,
                };
            }
        };

        // Debug: print output info
        #[cfg(debug_assertions)]
        {
            eprintln!("U2Netp: {} outputs", outputs.len());
            for (i, o) in outputs.iter().enumerate() {
                eprintln!(
                    "  output[{i}]: shape={:?}, dtype={:?}",
                    o.shape(),
                    o.datum_type()
                );
                if let Ok(s) = o.as_slice::<f32>() {
                    let min = s.iter().cloned().fold(f32::MAX, f32::min);
                    let max = s.iter().cloned().fold(f32::MIN, f32::max);
                    let mean = s.iter().sum::<f32>() / s.len() as f32;
                    eprintln!(
                        "    min={min:.6}, max={max:.6}, mean={mean:.6}, len={}",
                        s.len()
                    );
                }
            }
        }

        // Output 0: [1, 1, 320, 320] saliency map, values in [0, 1] (sigmoided)
        let raw = match outputs[0].as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => {
                return SaliencyMap {
                    data: vec![0.0; INPUT_SIZE * INPUT_SIZE],
                    width: INPUT_SIZE as u32,
                    height: INPUT_SIZE as u32,
                };
            }
        };

        // Min-max normalize to stretch contrast
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in raw {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let range = max_val - min_val;
        let data = if range > 1e-6 {
            raw.iter()
                .map(|&v| ((v - min_val) / range).clamp(0.0, 1.0))
                .collect()
        } else {
            vec![0.0; raw.len()]
        };

        SaliencyMap {
            data,
            width: INPUT_SIZE as u32,
            height: INPUT_SIZE as u32,
        }
    }
}
