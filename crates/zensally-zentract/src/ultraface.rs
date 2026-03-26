#![forbid(unsafe_code)]

//! UltraFace RFB-320 face detector via zentract plugin.
//!
//! Same model and pre/postprocessing as zensally-tract's UltraFace,
//! but inference runs through the zentract dlopen plugin.
//!
//! NOTE: zentract runs the full model on each `infer_raw()` call, even for
//! multi-output models. UltraFace has 2 outputs (scores, boxes), so we
//! call infer twice (~32ms instead of ~16ms). A future `zentract_infer_all`
//! API could halve this.

use std::cell::RefCell;

use zentract_api::{InferenceEngine, TensorMeta};
use zensally::preprocess::{Normalization, ResizeMode, preprocess_nchw};
use zensally::{FaceDetector, FaceRect, ImageRef};

/// Embedded gzip-compressed UltraFace RFB-320 ONNX model.
const MODEL_GZ: &[u8] = include_bytes!("../../zensally-tract/models/ultraface-rfb-320.onnx.gz");

const INPUT_W: usize = 320;
const INPUT_H: usize = 240;

/// Thread-local cached engine + loaded model handle.
struct CachedState {
    engine: InferenceEngine,
    handle_id: i64,
}

impl Drop for CachedState {
    fn drop(&mut self) {
        self.engine.free_raw(self.handle_id);
    }
}

thread_local! {
    static CACHE: RefCell<Option<CachedState>> = const { RefCell::new(None) };
}

fn ensure_loaded() -> Result<(), anyhow::Error> {
    CACHE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_some() {
            return Ok(());
        }
        let engine = crate::load_plugin()?;
        let model_bytes = crate::decompress_gz(MODEL_GZ);
        let input_meta = TensorMeta::f32_shape(&[1, 3, INPUT_H as u64, INPUT_W as u64]);
        let handle = engine.load_onnx(&model_bytes, input_meta)?;
        let handle_id = handle.into_raw();
        *opt = Some(CachedState { engine, handle_id });
        Ok(())
    })
}

fn infer_cached(input: &[f32], output_index: u32) -> Result<Vec<f32>, anyhow::Error> {
    CACHE.with(|cell| {
        let opt = cell.borrow();
        let state = opt.as_ref().expect("engine not loaded");
        let output = state.engine.infer_raw(state.handle_id, input, output_index)?;
        Ok(output.data)
    })
}

/// UltraFace RFB-320 face detector via zentract plugin.
pub struct UltraFaceDetector {
    score_threshold: f32,
    nms_iou_threshold: f32,
    preprocess_buf: Vec<f32>,
}

impl UltraFaceDetector {
    /// Create a new detector.
    ///
    /// Loads the zentract plugin and ONNX model on first call per thread.
    /// Returns an error with build instructions if the plugin is not found.
    pub fn new() -> Result<Self, anyhow::Error> {
        ensure_loaded()?;
        Ok(Self {
            score_threshold: 0.7,
            nms_iou_threshold: 0.3,
            preprocess_buf: vec![0.0f32; 3 * INPUT_H * INPUT_W],
        })
    }
}

impl FaceDetector for UltraFaceDetector {
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect> {
        let letterbox = preprocess_nchw(
            image.pixels,
            image.width,
            image.height,
            image.format,
            INPUT_W,
            INPUT_H,
            ResizeMode::Letterbox,
            Normalization::CenterScale,
            &mut self.preprocess_buf,
        );

        let input = &self.preprocess_buf[..3 * INPUT_H * INPUT_W];

        let scores = match infer_cached(input, 0) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let boxes = match infer_cached(input, 1) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        zensally::decode::decode_ultraface(
            &scores,
            &boxes,
            INPUT_W as f32,
            INPUT_H as f32,
            &letterbox,
            image.width as f32,
            image.height as f32,
            self.score_threshold,
            self.nms_iou_threshold,
        )
    }
}
