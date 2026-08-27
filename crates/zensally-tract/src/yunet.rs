#![forbid(unsafe_code)]

//! YuNet face detector (OpenCV Zoo 2023mar) via tract.
//!
//! Anchor-free detector with 3 strides (8, 16, 32).
//! Input: NCHW BGR float32 [0, 255] — no normalization.
//! Outputs: cls/obj (sigmoided), bbox (raw offsets), kps (landmarks) per stride.

use tract_onnx::prelude::*;
use zensally::{FaceDetector, FaceRect, ImageRef, PixelFormat};

/// Embedded gzip-compressed YuNet 2023mar ONNX model.
const MODEL_GZ: &[u8] = include_bytes!("../models/yunet_2023mar.onnx.gz");

/// Detection strides for the YuNet feature pyramid.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Default input size (width and height). Must be a multiple of 32.
/// The 2023mar ONNX has internal Resize ops hardcoded for 640x640.
const INPUT_W: usize = 640;
const INPUT_H: usize = 640;

/// Configuration for the YuNet detector.
#[derive(Debug, Clone)]
pub struct YuNetConfig {
    /// Minimum score (sqrt(cls * obj)) to keep a detection.
    pub score_threshold: f32,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f32,
}

impl Default for YuNetConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_iou_threshold: 0.3,
        }
    }
}

/// A raw detection in pixel coordinates of the padded input.
#[derive(Debug, Clone)]
struct RawDetection {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

fn iou(a: &RawDetection, b: &RawDetection) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn nms(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());
        for j in (i + 1)..detections.len() {
            if !suppressed[j] && iou(&detections[i], &detections[j]) >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// YuNet face detector using tract for pure-Rust ONNX inference.
pub struct YuNetDetector {
    model: Arc<TypedRunnableModel>,
    config: YuNetConfig,
    /// Reusable preprocessing buffer: [3, INPUT_H, INPUT_W] NCHW BGR.
    preprocess_buf: Vec<f32>,
    /// Output node indices for cls, obj, bbox at each stride.
    /// Order: [cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32]
    output_mapping: [usize; 9],
}

impl YuNetDetector {
    /// Create a new detector with default configuration.
    pub fn new() -> Result<Self, anyhow::Error> {
        Self::with_config(YuNetConfig::default())
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(config: YuNetConfig) -> Result<Self, anyhow::Error> {
        let model_bytes = crate::decompress_gz(MODEL_GZ);
        let onnx = tract_onnx::onnx();
        let mut model = onnx.model_for_read(&mut std::io::Cursor::new(&model_bytes))?;

        // Override input shape to our desired size
        model = model.with_input_fact(
            0,
            InferenceFact::dt_shape(DatumType::F32, [1, 3, INPUT_H, INPUT_W]),
        )?;

        // Map output names to indices before optimization.
        // YuNet outputs: cls_8, cls_16, cls_32, obj_8, obj_16, obj_32,
        //                bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
        let outlet_labels: Vec<(usize, String)> = model
            .output_outlets()?
            .iter()
            .enumerate()
            .filter_map(|(i, outlet)| {
                model
                    .outlet_label(*outlet)
                    .map(|name| (i, name.to_string()))
            })
            .collect();

        let find_output = |prefix: &str, stride: usize| -> usize {
            let name = format!("{prefix}_{stride}");
            outlet_labels
                .iter()
                .find(|(_, n)| n == &name)
                .unwrap_or_else(|| panic!("output {name} not found"))
                .0
        };

        let output_mapping = [
            find_output("cls", 8),
            find_output("cls", 16),
            find_output("cls", 32),
            find_output("obj", 8),
            find_output("obj", 16),
            find_output("obj", 32),
            find_output("bbox", 8),
            find_output("bbox", 16),
            find_output("bbox", 32),
        ];

        let model = model.into_optimized()?.into_runnable()?;

        let preprocess_buf = vec![0.0f32; 3 * INPUT_H * INPUT_W];

        Ok(Self {
            model,
            config,
            preprocess_buf,
            output_mapping,
        })
    }

    /// Preprocess: bilinear resize to INPUT_W x INPUT_H with letterboxing,
    /// BGR [0, 255] float32, NCHW layout.
    /// Returns (pad_left, pad_top, ratio).
    fn preprocess(
        &mut self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        format: PixelFormat,
    ) -> (f32, f32, f32) {
        let bpp = format.bytes_per_pixel();

        let ratio = (INPUT_W as f32 / src_w as f32).min(INPUT_H as f32 / src_h as f32);
        let resized_w = (src_w as f32 * ratio).round() as u32;
        let resized_h = (src_h as f32 * ratio).round() as u32;
        let pad_left = (INPUT_W as u32 - resized_w) / 2;
        let pad_top = (INPUT_H as u32 - resized_h) / 2;

        // Fill with zero (black padding)
        self.preprocess_buf.fill(0.0);

        let (r_idx, g_idx, b_idx) = match format {
            PixelFormat::Bgra => (2, 1, 0),
            PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
        };

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
        let plane_size = INPUT_H * INPUT_W;

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
                let pixel_idx = out_y * INPUT_W + out_x;

                // NCHW BGR layout, values in [0, 255]
                self.preprocess_buf[pixel_idx] = b;
                self.preprocess_buf[plane_size + pixel_idx] = g;
                self.preprocess_buf[2 * plane_size + pixel_idx] = r;
            }
        }

        (pad_left as f32, pad_top as f32, ratio)
    }

    /// Decode detections from model outputs.
    fn decode(
        &self,
        outputs: &[TValue],
        pad_left: f32,
        pad_top: f32,
        ratio: f32,
    ) -> Vec<RawDetection> {
        let mut detections = Vec::new();
        let threshold = self.config.score_threshold;

        for (stride_idx, &stride) in STRIDES.iter().enumerate() {
            let cls = match crate::output::plain_f32(&outputs[self.output_mapping[stride_idx]]) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let obj = match crate::output::plain_f32(&outputs[self.output_mapping[3 + stride_idx]])
            {
                Ok(s) => s,
                Err(_) => continue,
            };
            let bbox = match crate::output::plain_f32(&outputs[self.output_mapping[6 + stride_idx]])
            {
                Ok(s) => s,
                Err(_) => continue,
            };

            let cols = INPUT_W / stride;
            let rows = INPUT_H / stride;

            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;

                    let cls_score = cls[idx].clamp(0.0, 1.0);
                    let obj_score = obj[idx].clamp(0.0, 1.0);
                    let score = (cls_score * obj_score).sqrt();

                    if score < threshold {
                        continue;
                    }

                    let bi = idx * 4;
                    let cx = (col as f32 + bbox[bi]) * stride as f32;
                    let cy = (row as f32 + bbox[bi + 1]) * stride as f32;
                    let w = bbox[bi + 2].exp() * stride as f32;
                    let h = bbox[bi + 3].exp() * stride as f32;

                    // Convert from padded input space to original image space
                    let x = (cx - w * 0.5 - pad_left) / ratio;
                    let y = (cy - h * 0.5 - pad_top) / ratio;
                    let width = w / ratio;
                    let height = h / ratio;

                    detections.push(RawDetection {
                        x,
                        y,
                        width,
                        height,
                        confidence: score,
                    });
                }
            }
        }

        detections
    }
}

impl FaceDetector for YuNetDetector {
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect> {
        let (pad_left, pad_top, ratio) =
            self.preprocess(image.pixels, image.width, image.height, image.format);

        let input = match Tensor::from_shape(
            &[1, 3, INPUT_H, INPUT_W],
            &self.preprocess_buf[..3 * INPUT_H * INPUT_W],
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        let detections = self.decode(&outputs, pad_left, pad_top, ratio);
        let detections = nms(detections, self.config.nms_iou_threshold);

        let img_w = image.width as f32;
        let img_h = image.height as f32;

        detections
            .into_iter()
            .map(|d| FaceRect {
                x1: (d.x / img_w * 100.0).clamp(0.0, 100.0),
                y1: (d.y / img_h * 100.0).clamp(0.0, 100.0),
                x2: ((d.x + d.width) / img_w * 100.0).clamp(0.0, 100.0),
                y2: ((d.y + d.height) / img_h * 100.0).clamp(0.0, 100.0),
                confidence: d.confidence,
            })
            .collect()
    }
}
