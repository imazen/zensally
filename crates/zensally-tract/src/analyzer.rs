#![forbid(unsafe_code)]

//! High-level content analyzer combining UltraFace + MicroSalNet.
//!
//! Runs both detectors on a single image and returns a
//! [`ContentAnalysis`] for batch crop computation.

use zensally::crop::ContentAnalysis;
use zensally::{FaceDetector, ImageRef, SaliencyDetector};

use crate::microsalnet::MicroSalNet;
use crate::ultraface::UltraFaceDetector;

/// High-level content analyzer — UltraFace + MicroSalNet.
///
/// Runs face detection and saliency in sequence (~28ms total),
/// producing a [`ContentAnalysis`] that can compute crops for
/// multiple aspect ratios without re-running detection.
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
    pub fn analyze(&mut self, image: &ImageRef<'_>) -> ContentAnalysis {
        let faces = self.face_det.detect(image);
        let saliency = self.sal_det.saliency_map(image);
        ContentAnalysis {
            faces,
            saliency: Some(saliency),
        }
    }
}
