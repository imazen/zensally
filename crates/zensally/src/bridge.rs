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
    let mut focus_regions: Vec<FocusRect> = faces.into_iter().map(FocusRect::from).collect();

    for r in manual_focus {
        focus_regions.push(FocusRect::from(r));
    }

    SmartCropInput {
        focus_regions,
        heatmap: saliency.map(HeatMap::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_rect_to_focus_rect() {
        let face = FaceRect {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 40.0,
            confidence: 0.9,
        };
        let focus: FocusRect = face.into();
        assert_eq!(focus.x1, 10.0);
        assert_eq!(focus.y1, 20.0);
        assert_eq!(focus.x2, 30.0);
        assert_eq!(focus.y2, 40.0);
        assert_eq!(focus.weight, 0.9);
    }

    #[test]
    fn focus_region_to_focus_rect_full_weight() {
        let region = FocusRegion {
            x1: 5.0,
            y1: 10.0,
            x2: 95.0,
            y2: 90.0,
        };
        let focus: FocusRect = region.into();
        assert_eq!(focus.weight, 1.0);
    }

    #[test]
    fn saliency_map_to_heatmap() {
        let sal = SaliencyMap {
            data: vec![0.1, 0.9, 0.5, 0.7],
            width: 2,
            height: 2,
        };
        let hm: HeatMap = sal.into();
        assert_eq!(hm.width, 2);
        assert_eq!(hm.height, 2);
        assert_eq!(hm.data, vec![0.1, 0.9, 0.5, 0.7]);
    }

    #[test]
    fn build_input_merges_faces_and_manual() {
        let analysis = AnalysisOutput {
            faces: vec![FaceRect {
                x1: 10.0,
                y1: 10.0,
                x2: 30.0,
                y2: 30.0,
                confidence: 0.8,
            }],
            saliency: Some(SaliencyMap {
                data: vec![0.5; 4],
                width: 2,
                height: 2,
            }),
        };
        let manual = vec![FocusRegion {
            x1: 50.0,
            y1: 50.0,
            x2: 80.0,
            y2: 80.0,
        }];
        let input = build_smart_crop_input(analysis, &manual);
        assert_eq!(input.focus_regions.len(), 2);
        assert_eq!(input.focus_regions[0].weight, 0.8); // from face confidence
        assert_eq!(input.focus_regions[1].weight, 1.0); // manual = full weight
        assert!(input.heatmap.is_some());
    }

    #[test]
    fn build_input_no_saliency() {
        let analysis = AnalysisOutput {
            faces: Vec::new(),
            saliency: None,
        };
        let input = build_smart_crop_input(analysis, &[]);
        assert!(input.focus_regions.is_empty());
        assert!(input.heatmap.is_none());
    }
}
