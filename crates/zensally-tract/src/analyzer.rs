#![forbid(unsafe_code)]

//! High-level content analyzer combining UltraFace + MicroSalNet.
//!
//! Runs both detectors on a single image and returns detection results
//! that can be fed into `zenlayout::smart_crop::SmartCropInput` for
//! batch crop computation.

use zensally::{FaceDetector, FaceRect, ImageRef, SaliencyDetector, SaliencyMap};

use crate::microsalnet::MicroSalNet;
use crate::ultraface::UltraFaceDetector;

/// High-level content analyzer — UltraFace + MicroSalNet.
///
/// Runs face detection and saliency in sequence (~28ms total).
/// Convert the results to `zenlayout::smart_crop::SmartCropInput`
/// for content-aware cropping at multiple aspect ratios.
pub struct ContentAnalyzer {
    face_det: UltraFaceDetector,
    sal_det: MicroSalNet,
}

/// Detection results from content analysis.
///
/// Convert to `zenlayout::smart_crop::SmartCropInput` for crop computation:
/// ```ignore
/// use zenlayout::smart_crop::{FocusRect, HeatMap, SmartCropInput};
///
/// let result = analyzer.analyze(&image);
/// let input = SmartCropInput {
///     focus_regions: result.faces.into_iter().map(|f| FocusRect {
///         x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, weight: f.confidence,
///     }).collect(),
///     heatmap: result.saliency.map(|s| HeatMap {
///         data: s.data, width: s.width, height: s.height,
///     }),
/// };
/// ```
pub struct DetectionResult {
    /// Detected faces (percentage coordinates, sorted by confidence).
    pub faces: Vec<FaceRect>,
    /// Saliency heatmap at model resolution (128x128).
    pub saliency: Option<SaliencyMap>,
}

impl ContentAnalyzer {
    /// Create a new analyzer, loading both UltraFace and MicroSalNet models.
    pub fn new() -> Result<Self, anyhow::Error> {
        let face_det = UltraFaceDetector::new()?;
        let sal_det = MicroSalNet::new()?;
        Ok(Self { face_det, sal_det })
    }

    /// Run face detection + saliency on one image. ~28ms.
    pub fn analyze(&mut self, image: &ImageRef<'_>) -> DetectionResult {
        let faces = self.face_det.detect(image);
        let saliency = Some(self.sal_det.saliency_map(image));
        DetectionResult { faces, saliency }
    }
}
