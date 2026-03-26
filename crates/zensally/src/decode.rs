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
