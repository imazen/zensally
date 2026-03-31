#![forbid(unsafe_code)]

use crate::anchors::Anchor;

/// Variance values for box decoding (matching zineos BlazeFace).
const VARIANCE_XY: f32 = 0.1;
const VARIANCE_WH: f32 = 0.2;

/// A decoded detection before NMS, in pixel coordinates.
#[derive(Debug, Clone)]
pub struct RawDetection {
    /// Top-left x in pixel coordinates.
    pub x: f32,
    /// Top-left y in pixel coordinates.
    pub y: f32,
    /// Width in pixel coordinates.
    pub width: f32,
    /// Height in pixel coordinates.
    pub height: f32,
    /// Face confidence score.
    pub confidence: f32,
}

/// Decode a single box regression relative to its anchor.
///
/// Returns (x, y, width, height) in normalized coordinates (0..1).
#[inline]
fn decode_box(anchor: &Anchor, pred: &[f32]) -> (f32, f32, f32, f32) {
    let cx = anchor.cx + pred[0] * VARIANCE_XY * anchor.w;
    let cy = anchor.cy + pred[1] * VARIANCE_XY * anchor.h;
    let w = anchor.w * (pred[2] * VARIANCE_WH).exp();
    let h = anchor.h * (pred[3] * VARIANCE_WH).exp();
    (cx - w * 0.5, cy - h * 0.5, w, h)
}

/// Decode all detections above the score threshold.
///
/// - `boxes`: raw box regressions, shape [N, 4]
/// - `scores`: softmax scores, shape [N, 2] (index 1 = face)
/// - `anchors`: pre-computed anchor boxes
/// - `score_threshold`: minimum confidence to keep
/// - `scale_x`, `scale_y`: multiply normalized coords to get pixel coords
pub fn decode_detections(
    boxes: &[f32],
    scores: &[f32],
    anchors: &[Anchor],
    score_threshold: f32,
    scale_x: f32,
    scale_y: f32,
) -> Vec<RawDetection> {
    let n = anchors.len();
    let mut detections = Vec::new();

    for i in 0..n {
        let confidence = scores[i * 2 + 1]; // face score
        if confidence <= score_threshold {
            continue;
        }

        let pred = &boxes[i * 4..i * 4 + 4];
        let (x, y, w, h) = decode_box(&anchors[i], pred);

        detections.push(RawDetection {
            x: x * scale_x,
            y: y * scale_y,
            width: w * scale_x,
            height: h * scale_y,
            confidence,
        });
    }

    detections
}

/// Compute intersection-over-union between two detections.
#[inline]
fn iou(a: &RawDetection, b: &RawDetection) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Non-maximum suppression. Returns detections sorted by confidence (highest first).
pub fn nms(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    // Sort descending by confidence
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());
        for j in (i + 1)..detections.len() {
            if !suppressed[j] && iou(&detections[i], &detections[j]) >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
