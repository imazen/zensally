//! Non-maximum suppression for face detection.

use crate::FaceRect;

/// A raw detection in pixel coordinates (before percentage conversion).
#[derive(Debug, Clone)]
pub struct RawDetection {
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Detection confidence.
    pub confidence: f32,
}

/// Intersection-over-union of two detections.
pub fn iou(a: &RawDetection, b: &RawDetection) -> f32 {
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

/// Greedy NMS: keep highest-confidence detections, suppress overlapping ones.
pub fn nms(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(core::cmp::Ordering::Equal)
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

/// Convert raw pixel-space detections to percentage-coordinate [`FaceRect`]s.
pub fn to_face_rects(detections: Vec<RawDetection>, img_w: f32, img_h: f32) -> Vec<FaceRect> {
    detections
        .into_iter()
        .map(|d| FaceRect {
            x1: (d.x / img_w * 100.0).clamp(0.0, 100.0),
            y1: (d.y / img_h * 100.0).clamp(0.0, 100.0),
            x2: ((d.x + d.width) / img_w * 100.0).clamp(0.0, 100.0),
            y2: ((d.y + d.height) / img_h * 100.0).clamp(0.0, 100.0),
            confidence: d.confidence,
        })
        .collect()
}
