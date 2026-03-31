#![forbid(unsafe_code)]

//! MicroSalNet saliency detector via zentract plugin.
//!
//! Same model and pre/postprocessing as zensally-tract's MicroSalNet,
//! but inference runs through the zentract dlopen plugin.

use std::cell::RefCell;

use zensally::preprocess::{Normalization, ResizeMode, preprocess_nchw};
use zensally::{ImageRef, SaliencyDetector, SaliencyMap};
use zentract_api::{InferenceEngine, TensorMeta};

/// Embedded gzip-compressed MicroSalNet ONNX model.
const MODEL_GZ: &[u8] = include_bytes!("../../zensally-tract/models/microsalnet.onnx.gz");

const INPUT_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 128;

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
        let input_meta = TensorMeta::f32_shape(&[1, 3, INPUT_SIZE as u64, INPUT_SIZE as u64]);
        let handle = engine.load_onnx(&model_bytes, input_meta)?;
        let handle_id = handle.into_raw();
        *opt = Some(CachedState { engine, handle_id });
        Ok(())
    })
}

fn infer_cached(input: &[f32]) -> Result<Vec<f32>, anyhow::Error> {
    CACHE.with(|cell| {
        let opt = cell.borrow();
        let state = opt.as_ref().expect("engine not loaded");
        let output = state.engine.infer_raw(state.handle_id, input, 0)?;
        Ok(output.data)
    })
}

/// MicroSalNet saliency detector via zentract plugin.
pub struct MicroSalNet {
    preprocess_buf: Vec<f32>,
}

impl MicroSalNet {
    /// Create a new detector.
    ///
    /// Loads the zentract plugin and ONNX model on first call per thread.
    /// Returns an error with build instructions if the plugin is not found.
    pub fn new() -> Result<Self, anyhow::Error> {
        ensure_loaded()?;
        Ok(Self {
            preprocess_buf: vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE],
        })
    }
}

impl SaliencyDetector for MicroSalNet {
    fn saliency_map(&mut self, image: &ImageRef<'_>) -> SaliencyMap {
        preprocess_nchw(
            image.pixels,
            image.width,
            image.height,
            image.format,
            INPUT_SIZE,
            INPUT_SIZE,
            ResizeMode::Stretch,
            Normalization::UnitScale,
            &mut self.preprocess_buf,
        );

        let input = &self.preprocess_buf[..3 * INPUT_SIZE * INPUT_SIZE];

        let raw = match infer_cached(input) {
            Ok(v) => v,
            Err(_) => {
                return SaliencyMap {
                    data: vec![0.0; OUTPUT_SIZE * OUTPUT_SIZE],
                    width: OUTPUT_SIZE as u32,
                    height: OUTPUT_SIZE as u32,
                };
            }
        };

        zensally::decode::decode_microsalnet(&raw, OUTPUT_SIZE as u32, OUTPUT_SIZE as u32)
    }
}
