#![forbid(unsafe_code)]

//! MediaPipe BlazeFace front camera (128x128) detector via tract.
//!
//! Based on Google's BlazeFace paper and MediaPipe implementation.
//! Uses the PINTO0309 ONNX conversion of the original TFLite model.

use tract_onnx::prelude::*;
use zenfaces::{FaceDetector, FaceRect, ImageRef, PixelFormat};

/// Embedded MediaPipe BlazeFace front camera ONNX model.
const MODEL_BYTES: &[u8] = include_bytes!("../models/face_detection_front_128x128_float32.onnx");

const INPUT_SIZE: usize = 128;

/// MediaPipe BlazeFace anchor.
#[derive(Debug, Clone, Copy)]
struct Anchor {
    cx: f32,
    cy: f32,
}

/// Generate MediaPipe-style anchors.
///
/// Back camera uses strides [16, 32, 32, 32], front camera uses [8, 16, 16, 16].
/// Each layer has 2 anchors per spatial position (all size 1.0 — uniform).
fn generate_anchors(input_size: usize, strides: &[usize]) -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(896);

    for &stride in strides {
        let grid_size = input_size / stride;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let cx = (x as f32 + 0.5) * stride as f32;
                let cy = (y as f32 + 0.5) * stride as f32;
                // 2 anchors per cell, both at the same position
                anchors.push(Anchor { cx, cy });
                anchors.push(Anchor { cx, cy });
            }
        }
    }

    anchors
}

/// Configuration for the MediaPipe BlazeFace detector.
#[derive(Debug, Clone)]
pub struct MediaPipeBlazeFaceConfig {
    /// Minimum raw score (pre-sigmoid) to keep a detection.
    /// Default: 0.0 (sigmoid(0) = 0.5).
    pub raw_score_threshold: f32,
    /// Minimum confidence after sigmoid to report.
    pub min_confidence: f32,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f32,
}

impl Default for MediaPipeBlazeFaceConfig {
    fn default() -> Self {
        Self {
            raw_score_threshold: 1.0,
            min_confidence: 0.75,
            nms_iou_threshold: 0.3,
        }
    }
}

/// A raw detection in pixel coordinates (of the original image).
#[derive(Debug, Clone)]
struct RawDetection {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

    if union <= 0.0 { 0.0 } else { intersection / union }
}

fn nms(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] { continue; }
        keep.push(detections[i].clone());
        for j in (i + 1)..detections.len() {
            if !suppressed[j] && iou(&detections[i], &detections[j]) >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// MediaPipe BlazeFace front camera detector using tract.
pub struct MediaPipeBlazeFaceDetector {
    model: TypedRunnableModel<TypedModel>,
    anchors: Vec<Anchor>,
    config: MediaPipeBlazeFaceConfig,
    /// Reusable preprocessing buffer: [1, 128, 128, 3] NHWC.
    preprocess_buf: Vec<f32>,
}

impl MediaPipeBlazeFaceDetector {
    /// Create a new detector with default configuration.
    pub fn new() -> Result<Self, anyhow::Error> {
        Self::with_config(MediaPipeBlazeFaceConfig::default())
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(config: MediaPipeBlazeFaceConfig) -> Result<Self, anyhow::Error> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, INPUT_SIZE, INPUT_SIZE, 3]),
            )?
            .into_optimized()?
            .into_runnable()?;

        // Front camera: strides [8, 16, 16, 16]
        let anchors = generate_anchors(INPUT_SIZE, &[8, 16, 16, 16]);

        let preprocess_buf = vec![0.0f32; INPUT_SIZE * INPUT_SIZE * 3];

        Ok(Self {
            model,
            anchors,
            config,
            preprocess_buf,
        })
    }

    /// Preprocess: bilinear resize to 128x128 with letterboxing, RGB [-1, 1] normalized, NHWC.
    ///
    /// Returns (pad_left, pad_top, ratio).
    fn preprocess(
        &mut self,
        pixels: &[u8],
        src_w: u32,
        src_h: u32,
        format: PixelFormat,
    ) -> (f32, f32, f32) {
        let t = INPUT_SIZE as u32;
        let bpp = format.bytes_per_pixel();

        let ratio = (t as f32 / src_w as f32).min(t as f32 / src_h as f32);
        let resized_w = (src_w as f32 * ratio).round() as u32;
        let resized_h = (src_h as f32 * ratio).round() as u32;
        let pad_left = (t - resized_w) / 2;
        let pad_top = (t - resized_h) / 2;

        // Fill with zero (maps to -1.0 after normalization of black pixels... but
        // actually we want the letterbox to be neutral. Use 0.0 which is "gray" in [-1,1] space)
        self.preprocess_buf.fill(0.0);

        let (r_idx, g_idx, b_idx) = match format {
            PixelFormat::Bgra => (2, 1, 0),
            PixelFormat::Rgba | PixelFormat::Rgb => (0, 1, 2),
        };

        let x_ratio = if resized_w > 1 { (src_w as f32 - 1.0) / (resized_w as f32 - 1.0) } else { 0.0 };
        let y_ratio = if resized_h > 1 { (src_h as f32 - 1.0) / (resized_h as f32 - 1.0) } else { 0.0 };
        let src_stride = src_w as usize * bpp;
        let t_usize = INPUT_SIZE;

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
                // NHWC: [y, x, channel]
                let base = (out_y * t_usize + out_x) * 3;
                self.preprocess_buf[base] = r / 127.5 - 1.0;
                self.preprocess_buf[base + 1] = g / 127.5 - 1.0;
                self.preprocess_buf[base + 2] = b / 127.5 - 1.0;
            }
        }

        (pad_left as f32, pad_top as f32, ratio)
    }

    /// Decode outputs from the MediaPipe model.
    ///
    /// Outputs: scores1 [1, 512, 1], scores2 [1, 384, 1],
    ///          regressors1 [1, 512, 16], regressors2 [1, 384, 16]
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        scores1: &[f32],
        scores2: &[f32],
        regressors1: &[f32],
        regressors2: &[f32],
        pad_left: f32,
        pad_top: f32,
        ratio: f32,
    ) -> Vec<RawDetection> {
        let mut detections = Vec::new();
        let threshold = self.config.raw_score_threshold;

        // Process both output groups
        let groups: &[(&[f32], &[f32], usize)] = &[
            (scores1, regressors1, 0),
            (scores2, regressors2, 512),
        ];

        for &(scores, regressors, anchor_offset) in groups {
            let n = scores.len(); // number of anchors in this group
            for i in 0..n {
                let raw_score = scores[i];
                if raw_score < threshold {
                    continue;
                }

                let confidence = sigmoid(raw_score);
                if confidence < self.config.min_confidence {
                    continue;
                }

                let anchor = &self.anchors[anchor_offset + i];
                let reg = &regressors[i * 16..i * 16 + 4];

                // MediaPipe box decoding:
                // cx = anchor_cx + reg[0]
                // cy = anchor_cy + reg[1]
                // w = reg[2]
                // h = reg[3]
                // All in pixel coordinates of the input tensor (128x128)
                let cx = anchor.cx + reg[0];
                let cy = anchor.cy + reg[1];
                let w = reg[2];
                let h = reg[3];

                // Convert from 128x128 tensor space to original image space
                let x = (cx - w * 0.5 - pad_left) / ratio;
                let y = (cy - h * 0.5 - pad_top) / ratio;
                let width = w / ratio;
                let height = h / ratio;

                detections.push(RawDetection {
                    x,
                    y,
                    width,
                    height,
                    confidence,
                });
            }
        }

        detections
    }
}

impl FaceDetector for MediaPipeBlazeFaceDetector {
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect> {
        let (pad_left, pad_top, ratio) = self.preprocess(
            image.pixels,
            image.width,
            image.height,
            image.format,
        );

        // Build NHWC tensor [1, 128, 128, 3]
        let input = match Tensor::from_shape(
            &[1, INPUT_SIZE, INPUT_SIZE, 3],
            &self.preprocess_buf[..INPUT_SIZE * INPUT_SIZE * 3],
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Run inference
        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        // outputs: [scores1, scores2, regressors1, regressors2]
        let scores1 = match outputs[0].as_slice::<f32>() { Ok(s) => s, Err(_) => return Vec::new() };
        let scores2 = match outputs[1].as_slice::<f32>() { Ok(s) => s, Err(_) => return Vec::new() };
        let regressors1 = match outputs[2].as_slice::<f32>() { Ok(s) => s, Err(_) => return Vec::new() };
        let regressors2 = match outputs[3].as_slice::<f32>() { Ok(s) => s, Err(_) => return Vec::new() };

        let detections = self.decode(
            scores1, scores2, regressors1, regressors2,
            pad_left, pad_top, ratio,
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_count() {
        let anchors = generate_anchors(128, &[8, 16, 16, 16]);
        // stride 8: 16x16 * 2 = 512
        // stride 16: 8x8 * 2 = 128 (x3 layers) = 384
        // total = 896
        assert_eq!(anchors.len(), 896);
    }
}
