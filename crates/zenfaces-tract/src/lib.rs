#![forbid(unsafe_code)]

mod anchors;
mod decode;
mod preprocess;

use anchors::{generate_anchors, Anchor, BLAZEFACE_ANCHOR_PARAMS};
use decode::{decode_detections, nms};
use preprocess::preprocess;

use tract_onnx::prelude::*;
use zenfaces::{FaceDetector, FaceRect, ImageRef};

/// Embedded BlazeFace-320 ONNX model.
const MODEL_BYTES: &[u8] = include_bytes!("../models/blazeface-320.onnx");

const TARGET_SIZE: u32 = 320;

/// Configuration for the BlazeFace detector.
#[derive(Debug, Clone)]
pub struct BlazeFaceConfig {
    /// Minimum confidence score to keep a detection (0.0–1.0).
    pub score_threshold: f32,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f32,
}

impl Default for BlazeFaceConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.95,
            nms_iou_threshold: 0.3,
        }
    }
}

/// BlazeFace face detector using tract for pure-Rust ONNX inference.
pub struct BlazeFaceDetector {
    model: TypedRunnableModel<TypedModel>,
    anchors: Vec<Anchor>,
    config: BlazeFaceConfig,
    /// Reusable preprocessing buffer: [3, TARGET_SIZE, TARGET_SIZE].
    preprocess_buf: Vec<f32>,
}

impl BlazeFaceDetector {
    /// Create a new detector with default configuration.
    pub fn new() -> Result<Self, anyhow::Error> {
        Self::with_config(BlazeFaceConfig::default())
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(config: BlazeFaceConfig) -> Result<Self, anyhow::Error> {
        let t = TARGET_SIZE as usize;

        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, &[1, 3, t, t]),
            )?
            .into_optimized()?
            .into_runnable()?;

        let anchors = generate_anchors(&BLAZEFACE_ANCHOR_PARAMS, (t, t));
        let preprocess_buf = vec![0.0f32; 3 * t * t];

        Ok(Self {
            model,
            anchors,
            config,
            preprocess_buf,
        })
    }
}

impl FaceDetector for BlazeFaceDetector {
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect> {
        let t = TARGET_SIZE as usize;

        // Preprocess: resize + letterbox to 320x320, BGR, mean subtract
        let prep = preprocess(
            image.pixels,
            image.width,
            image.height,
            image.format,
            TARGET_SIZE,
            &mut self.preprocess_buf,
        );

        // Build input tensor [1, 3, 320, 320]
        let input = match Tensor::from_shape(&[1, 3, t, t], &self.preprocess_buf[..3 * t * t]) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Run inference
        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        // Extract output tensors
        let boxes = match outputs[0].as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let scores = match outputs[1].as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        // Decode: anchors produce normalized coords in the 320x320 tensor space.
        // We decode to pixel coords in the 320x320 space, then map back to original.
        let detections = decode_detections(
            boxes,
            scores,
            &self.anchors,
            self.config.score_threshold,
            TARGET_SIZE as f32, // scale_x: normalized → 320px
            TARGET_SIZE as f32, // scale_y: normalized → 320px
        );

        let detections = nms(detections, self.config.nms_iou_threshold);

        // Map from 320x320 tensor space back to original image coordinates
        let img_w = image.width as f32;
        let img_h = image.height as f32;
        let pad_left = prep.pad_left as f32;
        let pad_top = prep.pad_top as f32;
        let ratio = prep.ratio;

        detections
            .into_iter()
            .map(|d| {
                // Remove letterbox padding, then undo resize
                let x = (d.x - pad_left) / ratio;
                let y = (d.y - pad_top) / ratio;
                let w = d.width / ratio;
                let h = d.height / ratio;

                FaceRect {
                    x1: (x / img_w * 100.0).clamp(0.0, 100.0),
                    y1: (y / img_h * 100.0).clamp(0.0, 100.0),
                    x2: ((x + w) / img_w * 100.0).clamp(0.0, 100.0),
                    y2: ((y + h) / img_h * 100.0).clamp(0.0, 100.0),
                    confidence: d.confidence,
                }
            })
            .collect()
    }
}
