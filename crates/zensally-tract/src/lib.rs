#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "blazeface320")]
mod anchors;
#[cfg(feature = "blazeface320")]
mod decode;
#[cfg(feature = "blazeface320")]
#[doc(hidden)]
pub mod preprocess;

#[cfg(feature = "mediapipe")]
pub mod mediapipe;

#[cfg(feature = "yunet")]
pub mod yunet;

#[cfg(feature = "ultraface")]
pub mod ultraface;

#[cfg(feature = "u2netp")]
pub mod u2netp;

#[cfg(feature = "blazeface320")]
use anchors::{generate_anchors, Anchor, BLAZEFACE_ANCHOR_PARAMS};
#[cfg(feature = "blazeface320")]
use decode::{decode_detections, nms};
#[cfg(feature = "blazeface320")]
use preprocess::preprocess;

#[cfg(feature = "blazeface320")]
use tract_onnx::prelude::*;
#[cfg(feature = "blazeface320")]
use zensally::{FaceDetector, FaceRect, ImageRef};

#[cfg(feature = "mediapipe")]
pub use mediapipe::{MediaPipeBlazeFaceConfig, MediaPipeBlazeFaceDetector};

#[cfg(feature = "yunet")]
pub use yunet::{YuNetConfig, YuNetDetector};

// UltraFace is the recommended default: 85% recall at 16ms on WIDER FACE.
#[cfg(feature = "ultraface")]
pub use ultraface::{UltraFaceConfig, UltraFaceDetector};

#[cfg(feature = "u2netp")]
pub use u2netp::U2NetpDetector;

/// Decompress a gzip-compressed model embedded via `include_bytes!`.
///
/// Reads the original size from the gzip ISIZE trailer (last 4 bytes).
pub(crate) fn decompress_gz(compressed: &[u8]) -> alloc::vec::Vec<u8> {
    let len = compressed.len();
    let orig_size = u32::from_le_bytes([
        compressed[len - 4],
        compressed[len - 3],
        compressed[len - 2],
        compressed[len - 1],
    ]) as usize;

    let mut decompressor = zenflate::Decompressor::new();
    let mut output = alloc::vec![0u8; orig_size];
    let outcome = decompressor
        .gzip_decompress(compressed, &mut output, enough::Unstoppable)
        .expect("embedded model decompression failed");
    debug_assert_eq!(outcome.output_written, orig_size);
    output
}

/// Embedded gzip-compressed zineos BlazeFace-320 ONNX model.
#[cfg(feature = "blazeface320")]
const MODEL_GZ: &[u8] = include_bytes!("../models/blazeface-320.onnx.gz");

#[cfg(feature = "blazeface320")]
const TARGET_SIZE: u32 = 320;

/// Configuration for the zineos BlazeFace-320 detector.
#[cfg(feature = "blazeface320")]
#[derive(Debug, Clone)]
pub struct BlazeFaceConfig {
    /// Minimum confidence score to keep a detection (0.0–1.0).
    pub score_threshold: f32,
    /// IoU threshold for non-maximum suppression.
    pub nms_iou_threshold: f32,
}

#[cfg(feature = "blazeface320")]
impl Default for BlazeFaceConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.95,
            nms_iou_threshold: 0.3,
        }
    }
}

/// zineos BlazeFace-320 face detector using tract for pure-Rust ONNX inference.
///
/// This is the heavier RetinaFace-style variant (~69ms). For better performance,
/// prefer [`UltraFaceDetector`] (~16ms, higher recall).
#[cfg(feature = "blazeface320")]
pub struct BlazeFaceDetector {
    model: TypedRunnableModel<TypedModel>,
    anchors: Vec<Anchor>,
    config: BlazeFaceConfig,
    preprocess_buf: Vec<f32>,
}

#[cfg(feature = "blazeface320")]
impl BlazeFaceDetector {
    pub fn new() -> Result<Self, anyhow::Error> {
        Self::with_config(BlazeFaceConfig::default())
    }

    pub fn with_config(config: BlazeFaceConfig) -> Result<Self, anyhow::Error> {
        let t = TARGET_SIZE as usize;

        let model_bytes = decompress_gz(MODEL_GZ);
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&model_bytes))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::F32, [1, 3, t, t]),
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

#[cfg(feature = "blazeface320")]
impl FaceDetector for BlazeFaceDetector {
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect> {
        let t = TARGET_SIZE as usize;

        let prep = preprocess(
            image.pixels,
            image.width,
            image.height,
            image.format,
            TARGET_SIZE,
            &mut self.preprocess_buf,
        );

        let input = match Tensor::from_shape(&[1, 3, t, t], &self.preprocess_buf[..3 * t * t]) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let outputs = match self.model.run(tvec!(input.into())) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        let boxes = match outputs[0].as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let scores = match outputs[1].as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let detections = decode_detections(
            boxes,
            scores,
            &self.anchors,
            self.config.score_threshold,
            TARGET_SIZE as f32,
            TARGET_SIZE as f32,
        );

        let detections = nms(detections, self.config.nms_iou_threshold);

        let img_w = image.width as f32;
        let img_h = image.height as f32;
        let pad_left = prep.pad_left as f32;
        let pad_top = prep.pad_top as f32;
        let ratio = prep.ratio;

        detections
            .into_iter()
            .map(|d| {
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
