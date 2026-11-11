#![forbid(unsafe_code)]

//! High-level content analyzers.
//!
//! [`ContentAnalyzer`] (v0.1): UltraFace + MicroSalNet, feeding
//! `zenlayout::smart_crop::SmartCropInput` via `zensally::bridge`.
//!
//! [`CropAnalyzerV2`] (v0.2): YuNet + 5-point landmarks, MicroSalNet
//! v3ds-w24 and SelfieSeg fused by `zensally::stream_prep` +
//! `zensally::sat_crop` into a reusable `DualIntegralMap` for
//! single-pass multi-aspect crop solving.

use zensally::ImageRef;

#[cfg(feature = "ultraface")]
use zensally::{AnalysisOutput, FaceDetector, SaliencyDetector};

#[cfg(feature = "ultraface")]
use crate::microsalnet::MicroSalNet;
#[cfg(feature = "ultraface")]
use crate::ultraface::UltraFaceDetector;

/// High-level content analyzer — UltraFace + MicroSalNet.
///
/// Runs face detection and saliency in sequence (~28ms total).
/// Use [`AnalysisOutput`] with `zensally::bridge::build_smart_crop_input()`
/// for content-aware cropping at multiple aspect ratios.
#[cfg(feature = "ultraface")]
pub struct ContentAnalyzer {
    face_det: UltraFaceDetector,
    sal_det: MicroSalNet,
}

#[cfg(feature = "ultraface")]
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

/// Analysis target for [`CropAnalyzerV2`]: 512×384 pixels over a stride-8
/// 64×48 feature grid (matches the v0.2 corpus evaluation geometry).
const TARGET_W: usize = 512;
const TARGET_H: usize = 384;
const GRID_W: usize = TARGET_W / 8;
const GRID_H: usize = TARGET_H / 8;

/// v0.2 crop analyzer — YuNet(+5kps) + MicroSalNet v3ds-w24 + SelfieSeg
/// driving the fused `stream_prep` → `DualIntegralMap` → `solve_crop_sat`
/// pipeline.
///
/// Detection runs once per image; the resulting integral map is reused
/// across every requested aspect/mode pair via [`Self::compute_crops`].
/// Measured on the 741-photo corpus evaluation: ~5.4 ms `stream_prep`,
/// ~0.4 ms map build, ~0.4 ms for all eight solves on a mid-range CPU.
#[cfg(all(feature = "yunet", feature = "microsalnet", feature = "selfie_seg"))]
pub struct CropAnalyzerV2 {
    face_det: crate::yunet::YuNetDetector,
    sal_det: crate::microsalnet::MicroSalNetV3,
    seg_det: crate::selfie_seg::SelfieSeg,
    /// Reusable NHWC output: `3 * TARGET_W * TARGET_H`.
    nhwc: Vec<f32>,
    /// Reusable stride-8 classical energy: `GRID_W * GRID_H`.
    classical: Vec<f32>,
    /// Reusable per-grid-row horizon profile: `GRID_H`.
    horizon: Vec<f32>,
    /// Reusable prep scratch (`prep_scratch_len` elements).
    scratch: Vec<f32>,
    prep_cfg: zensally::preprocess::PreprocessConfig,
}

#[cfg(all(feature = "yunet", feature = "microsalnet", feature = "selfie_seg"))]
impl CropAnalyzerV2 {
    /// Load the YuNet, MicroSalNet v3ds-w24 and SelfieSeg models and
    /// allocate the reusable prep buffers.
    pub fn new() -> Result<Self, anyhow::Error> {
        use zensally::preprocess::{Normalization, ResizeMode};
        Ok(Self {
            face_det: crate::yunet::YuNetDetector::new()?,
            sal_det: crate::microsalnet::MicroSalNetV3::new()?,
            seg_det: crate::selfie_seg::SelfieSeg::new()?,
            nhwc: vec![0.0; 3 * TARGET_W * TARGET_H],
            classical: vec![0.0; GRID_W * GRID_H],
            horizon: vec![0.0; GRID_H],
            scratch: vec![0.0; zensally::stream_prep::prep_scratch_len(TARGET_W, GRID_W, GRID_H)],
            prep_cfg: zensally::preprocess::PreprocessConfig {
                target_w: TARGET_W,
                target_h: TARGET_H,
                mode: ResizeMode::Stretch,
                norm: Normalization::UnitScale,
            },
        })
    }

    /// Run detection and fused preprocessing, then build the
    /// [`DualIntegralMap`] and its grid→source mapping.
    ///
    /// The map and `to_src` are independent of the requested aspect/mode —
    /// build once, then call [`zensally::sat_crop::solve_crops_sat`] (or
    /// [`Self::compute_crop`]/[`Self::compute_crops`]) for any number of
    /// crop requests.
    pub fn build_integral_map(
        &mut self,
        image: &ImageRef<'_>,
        manual_focus: &[zensally::FocusRegion],
    ) -> (
        zensally::sat_crop::DualIntegralMap,
        zensally::sat_crop::TargetToSrc,
    ) {
        use zensally::SaliencyDetector;
        use zensally::sat_crop::{CropInputSources, DualIntegralMap, TargetToSrc};
        use zensally::stream_prep::{self, PrepOutputs};

        let faces = self.face_det.detect_with_landmarks(image);
        let saliency = self.sal_det.saliency_map(image);
        let matte = self.seg_det.saliency_map(image);

        let mut outputs = PrepOutputs {
            output_nhwc: &mut self.nhwc,
            classical_energy: &mut self.classical,
            horizon_profile: &mut self.horizon,
            scratch: &mut self.scratch,
        };
        stream_prep::preprocess_nhwc_strided(
            image.pixels,
            image.width,
            image.height,
            image.width as usize * image.format.bytes_per_pixel(),
            image.format,
            &self.prep_cfg,
            &mut outputs,
        );

        let to_src =
            TargetToSrc::stretch(image.width, image.height, TARGET_W as u32, TARGET_H as u32);
        let sources = CropInputSources {
            saliency: Some(&saliency),
            matte: Some(&matte),
            faces: &faces,
            manual_focus,
            classical_energy: Some(&self.classical),
            horizon_profile: Some(&self.horizon),
        };
        (
            DualIntegralMap::from_crop_inputs(&sources, GRID_W, GRID_H, &to_src, 0.25),
            to_src,
        )
    }

    /// Analyze the image and solve one `(aspect, mode)` crop.
    /// Returns `None` for degenerate inputs (e.g. zero-size aspect).
    pub fn compute_crop(
        &mut self,
        image: &ImageRef<'_>,
        target_aspect: (u32, u32),
        mode: zensally::sat_crop::SatCropMode,
        manual_focus: &[zensally::FocusRegion],
    ) -> Option<zensally::CropRect> {
        let (map, to_src) = self.build_integral_map(image, manual_focus);
        zensally::sat_crop::solve_crop_sat(
            &map,
            &to_src,
            target_aspect,
            mode,
            &zensally::sat_crop::SatCropConfig::default(),
        )
    }

    /// Analyze the image once, then solve every `(aspect, mode)` request
    /// against the shared integral map. Results land in request order.
    pub fn compute_crops(
        &mut self,
        image: &ImageRef<'_>,
        requests: &[((u32, u32), zensally::sat_crop::SatCropMode)],
        manual_focus: &[zensally::FocusRegion],
    ) -> Vec<Option<zensally::CropRect>> {
        let (map, to_src) = self.build_integral_map(image, manual_focus);
        zensally::sat_crop::solve_crops_sat(
            &map,
            &to_src,
            requests,
            &zensally::sat_crop::SatCropConfig::default(),
        )
    }
}
