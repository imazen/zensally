#![forbid(unsafe_code)]

/// Pre-computed anchor box in normalized coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Anchor generation parameters matching the zineos BlazeFace variant.
pub struct AnchorParams {
    pub min_sizes: &'static [&'static [usize]],
    pub steps: &'static [usize],
}

/// Default parameters for BlazeFace-320.
pub const BLAZEFACE_ANCHOR_PARAMS: AnchorParams = AnchorParams {
    min_sizes: &[&[8, 11], &[14, 19, 26, 38, 64, 149]],
    steps: &[8, 16],
};

/// Generate anchors for a given image size.
///
/// `image_size` is `(width, height)` of the input tensor spatial dimensions.
pub fn generate_anchors(params: &AnchorParams, image_size: (usize, usize)) -> Vec<Anchor> {
    let (img_w, img_h) = image_size;

    let feature_maps: Vec<(usize, usize)> = params
        .steps
        .iter()
        .map(|&step| (img_w / step, img_h / step))
        .collect();

    let mut anchors = Vec::new();

    for ((fm, min_sizes), &step) in feature_maps.iter().zip(params.min_sizes.iter()).zip(params.steps.iter()) {
        let (fm_w, fm_h) = *fm;
        // Outer loop: rows (y), inner loop: columns (x), matching rust-faces ordering
        for row in 0..fm_h {
            for col in 0..fm_w {
                for &min_size in *min_sizes {
                    let cx = (col as f32 + 0.5) * step as f32 / img_w as f32;
                    let cy = (row as f32 + 0.5) * step as f32 / img_h as f32;
                    let w = min_size as f32 / img_w as f32;
                    let h = min_size as f32 / img_h as f32;
                    anchors.push(Anchor { cx, cy, w, h });
                }
            }
        }
    }

    anchors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_count_320x320() {
        let anchors = generate_anchors(&BLAZEFACE_ANCHOR_PARAMS, (320, 320));
        // stride 8: 40*40*2 = 3200, stride 16: 20*20*6 = 2400, total = 5600
        assert_eq!(anchors.len(), 5600);
    }

    #[test]
    fn first_anchor_position() {
        let anchors = generate_anchors(&BLAZEFACE_ANCHOR_PARAMS, (320, 320));
        let a = &anchors[0];
        // First anchor: col=0, row=0, step=8, min_size=8
        let expected_cx = 0.5 * 8.0 / 320.0;
        let expected_cy = 0.5 * 8.0 / 320.0;
        assert!((a.cx - expected_cx).abs() < 1e-6);
        assert!((a.cy - expected_cy).abs() < 1e-6);
        assert!((a.w - 8.0 / 320.0).abs() < 1e-6);
    }
}
