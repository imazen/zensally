//! Conversions from zensally types to [`zenlayout::smart_crop`] types.
//!
//! Enabled by the `zenlayout` feature.

use zenlayout::smart_crop::{FocusRect, HeatMap, SmartCropInput};

use crate::{AnalysisOutput, FaceRect, FocusRegion, SaliencyMap};

impl From<FaceRect> for FocusRect {
    fn from(f: FaceRect) -> Self {
        FocusRect {
            x1: f.x1,
            y1: f.y1,
            x2: f.x2,
            y2: f.y2,
            weight: f.confidence,
        }
    }
}

impl From<&FaceRect> for FocusRect {
    fn from(f: &FaceRect) -> Self {
        FocusRect {
            x1: f.x1,
            y1: f.y1,
            x2: f.x2,
            y2: f.y2,
            weight: f.confidence,
        }
    }
}

impl From<FocusRegion> for FocusRect {
    fn from(r: FocusRegion) -> Self {
        FocusRect {
            x1: r.x1,
            y1: r.y1,
            x2: r.x2,
            y2: r.y2,
            weight: 1.0,
        }
    }
}

impl From<&FocusRegion> for FocusRect {
    fn from(r: &FocusRegion) -> Self {
        FocusRect {
            x1: r.x1,
            y1: r.y1,
            x2: r.x2,
            y2: r.y2,
            weight: 1.0,
        }
    }
}

impl From<SaliencyMap> for HeatMap {
    fn from(s: SaliencyMap) -> Self {
        HeatMap {
            data: s.data,
            width: s.width,
            height: s.height,
        }
    }
}

/// Build a [`SmartCropInput`] from an [`AnalysisOutput`] and optional
/// manual focus regions.
pub fn build_smart_crop_input(
    analysis: AnalysisOutput,
    manual_focus: &[FocusRegion],
) -> SmartCropInput {
    let mut focus_regions: Vec<FocusRect> =
        analysis.faces.into_iter().map(FocusRect::from).collect();

    // Manual focus regions get full weight.
    for r in manual_focus {
        focus_regions.push(FocusRect::from(r));
    }

    SmartCropInput {
        focus_regions,
        heatmap: analysis.saliency.map(HeatMap::from),
    }
}

/// Build a [`SmartCropInput`] from raw components.
pub fn build_smart_crop_input_raw(
    faces: Vec<FaceRect>,
    saliency: Option<SaliencyMap>,
    manual_focus: &[FocusRegion],
) -> SmartCropInput {
    let mut focus_regions: Vec<FocusRect> =
        faces.into_iter().map(FocusRect::from).collect();

    for r in manual_focus {
        focus_regions.push(FocusRect::from(r));
    }

    SmartCropInput {
        focus_regions,
        heatmap: saliency.map(HeatMap::from),
    }
}
