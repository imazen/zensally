#![forbid(unsafe_code)]

//! High-level content analyzer combining UltraFace + MicroSalNet.
//!
//! Runs both detectors on a single image and returns detection results
//! that can be fed into `zenlayout::smart_crop::SmartCropInput` for
//! batch crop computation.

use zensally::{AnalysisOutput, FaceDetector, ImageRef, SaliencyDetector};

use crate::microsalnet::MicroSalNet;
use crate::ultraface::UltraFaceDetector;

/// High-level content analyzer — UltraFace + MicroSalNet.
///
/// Runs face detection and saliency in sequence (~28ms total).
/// Use [`AnalysisOutput`] with `zensally::bridge::build_smart_crop_input()`
/// for content-aware cropping at multiple aspect ratios.
pub struct ContentAnalyzer {
    face_det: UltraFaceDetector,
    sal_det: MicroSalNet,
}

impl ContentAnalyzer {
    /// Create a new analyzer, loading both UltraFace and MicroSalNet models.
    pub fn new() -> Result<Self, anyhow::Error> {
        let face_det = UltraFaceDetector::new()?;
        let sal_det = MicroSalNet::new()?;
        Ok(Self { face_det, sal_det })
    }

    /// Run face detection + saliency on one image. ~28ms.
    pub fn analyze(&mut self, image: &ImageRef<'_>) -> AnalysisOutput {
        let faces = self.face_det.detect(image);
        let saliency = Some(self.sal_det.saliency_map(image));
        AnalysisOutput { faces, saliency }
    }
}
