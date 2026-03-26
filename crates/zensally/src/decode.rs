//! Output decoding for specific ONNX model architectures.
//!
//! Converts raw model output tensors to zensally types ([`FaceRect`], [`SaliencyMap`]).

use crate::FaceRect;
use crate::SaliencyMap;
use crate::nms::{RawDetection, nms, to_face_rects};
use crate::preprocess::LetterboxInfo;

/// Decode UltraFace RFB-320 output tensors to face detections.
///
/// # Arguments
/// * `scores` — flat f32 slice of shape `[1, N, 2]` (background, face per anchor)
/// * `boxes` — flat f32 slice of shape `[1, N, 4]` (xmin, ymin, xmax, ymax, normalized 0-1)
/// * `input_w` — model input width (320)
/// * `input_h` — model input height (240)
/// * `letterbox` — letterbox info from preprocessing
/// * `img_w` — original image width in pixels
/// * `img_h` — original image height in pixels
/// * `score_threshold` — minimum confidence to keep
/// * `nms_iou_threshold` — IoU threshold for NMS
#[allow(clippy::too_many_arguments)]
pub fn decode_ultraface(
    scores: &[f32],
    boxes: &[f32],
    input_w: f32,
    input_h: f32,
    letterbox: &LetterboxInfo,
    img_w: f32,
    img_h: f32,
    score_threshold: f32,
    nms_iou_threshold: f32,
) -> Vec<FaceRect> {
    let n_anchors = scores.len() / 2;
    let mut detections = Vec::new();

    for i in 0..n_anchors {
        let face_score = scores[i * 2 + 1];
        if face_score < score_threshold {
            continue;
        }

        let bi = i * 4;
        let xmin = boxes[bi] * input_w;
        let ymin = boxes[bi + 1] * input_h;
        let xmax = boxes[bi + 2] * input_w;
        let ymax = boxes[bi + 3] * input_h;

        // Reverse letterbox: padded input space → original image space
        let x = (xmin - letterbox.pad_left) / letterbox.ratio;
        let y = (ymin - letterbox.pad_top) / letterbox.ratio;
        let w = (xmax - xmin) / letterbox.ratio;
        let h = (ymax - ymin) / letterbox.ratio;

        detections.push(RawDetection {
            x,
            y,
            width: w,
            height: h,
            confidence: face_score,
        });
    }

    let detections = nms(detections, nms_iou_threshold);
    to_face_rects(detections, img_w, img_h)
}

/// Decode MicroSalNet output tensor to a saliency map.
///
/// # Arguments
/// * `raw` — flat f32 slice of shape `[1, 1, H, W]` (sigmoided [0, 1])
/// * `output_w` — expected output width (128)
/// * `output_h` — expected output height (128)
pub fn decode_microsalnet(raw: &[f32], output_w: u32, output_h: u32) -> SaliencyMap {
    let data: Vec<f32> = raw.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    SaliencyMap {
        data,
        width: output_w,
        height: output_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::LetterboxInfo;

    #[test]
    fn decode_ultraface_no_detections_below_threshold() {
        // 2 anchors, both below threshold
        let scores = [0.9, 0.1, 0.8, 0.2]; // [bg, face] per anchor
        let boxes = [0.1, 0.1, 0.5, 0.5, 0.6, 0.6, 0.9, 0.9];
        let lb = LetterboxInfo { ratio: 1.0, pad_left: 0.0, pad_top: 0.0 };
        let result = decode_ultraface(&scores, &boxes, 320.0, 240.0, &lb, 320.0, 240.0, 0.7, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn decode_ultraface_single_detection() {
        // 1 anchor, face score = 0.95
        let scores = [0.05, 0.95];
        // Box: xmin=0.25, ymin=0.25, xmax=0.75, ymax=0.75 (normalized 0-1)
        let boxes = [0.25, 0.25, 0.75, 0.75];
        let lb = LetterboxInfo { ratio: 1.0, pad_left: 0.0, pad_top: 0.0 };
        let result = decode_ultraface(&scores, &boxes, 320.0, 240.0, &lb, 320.0, 240.0, 0.5, 0.3);
        assert_eq!(result.len(), 1);
        let r = &result[0];
        assert!((r.confidence - 0.95).abs() < 1e-6);
        // xmin=0.25*320=80, xmax=0.75*320=240 → x1=80/320*100=25%, x2=75%
        assert!((r.x1 - 25.0).abs() < 1.0);
        assert!((r.x2 - 75.0).abs() < 1.0);
    }

    #[test]
    fn decode_ultraface_with_letterbox() {
        let scores = [0.0, 0.9];
        let boxes = [0.25, 0.25, 0.75, 0.75];
        // ratio=0.5, pad_left=80, pad_top=0: image was shrunk to half
        let lb = LetterboxInfo { ratio: 0.5, pad_left: 80.0, pad_top: 0.0 };
        let result = decode_ultraface(&scores, &boxes, 320.0, 240.0, &lb, 640.0, 480.0, 0.5, 0.3);
        assert_eq!(result.len(), 1);
        // After letterbox reversal, coordinates should map back to original image
    }

    #[test]
    fn decode_microsalnet_clamps_values() {
        let raw = [-0.5, 0.5, 1.5, 0.8];
        let map = decode_microsalnet(&raw, 2, 2);
        assert_eq!(map.width, 2);
        assert_eq!(map.height, 2);
        assert_eq!(map.data, vec![0.0, 0.5, 1.0, 0.8]);
    }

    #[test]
    fn decode_microsalnet_preserves_dims() {
        let raw = vec![0.5; 128 * 128];
        let map = decode_microsalnet(&raw, 128, 128);
        assert_eq!(map.data.len(), 128 * 128);
        assert_eq!(map.width, 128);
        assert_eq!(map.height, 128);
    }
}
