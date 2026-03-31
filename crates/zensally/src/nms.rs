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

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, c: f32) -> RawDetection {
        RawDetection {
            x,
            y,
            width: w,
            height: h,
            confidence: c,
        }
    }

    #[test]
    fn iou_identical_boxes() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = det(20.0, 20.0, 10.0, 10.0, 1.0);
        assert!(iou(&a, &b) < 1e-6);
    }

    #[test]
    fn iou_partial_overlap() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = det(5.0, 0.0, 10.0, 10.0, 1.0);
        // overlap: 5x10=50, union: 100+100-50=150
        let expected = 50.0 / 150.0;
        assert!((iou(&a, &b) - expected).abs() < 1e-4);
    }

    #[test]
    fn nms_suppresses_overlapping() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9),
            det(1.0, 1.0, 10.0, 10.0, 0.8),   // high IoU with first
            det(50.0, 50.0, 10.0, 10.0, 0.7), // no overlap
        ];
        let kept = nms(dets, 0.3);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
        assert!((kept[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_empty_input() {
        assert!(nms(Vec::new(), 0.5).is_empty());
    }

    #[test]
    fn nms_single_detection() {
        let dets = vec![det(0.0, 0.0, 10.0, 10.0, 0.9)];
        assert_eq!(nms(dets, 0.5).len(), 1);
    }

    #[test]
    fn to_face_rects_converts_correctly() {
        let dets = vec![det(10.0, 20.0, 30.0, 40.0, 0.95)];
        let rects = to_face_rects(dets, 100.0, 200.0);
        assert_eq!(rects.len(), 1);
        let r = &rects[0];
        assert!((r.x1 - 10.0).abs() < 1e-4); // 10/100*100
        assert!((r.y1 - 10.0).abs() < 1e-4); // 20/200*100
        assert!((r.x2 - 40.0).abs() < 1e-4); // (10+30)/100*100
        assert!((r.y2 - 30.0).abs() < 1e-4); // (20+40)/200*100
        assert!((r.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn to_face_rects_clamps_to_bounds() {
        // Detection extending outside image
        let dets = vec![det(-10.0, -10.0, 200.0, 200.0, 0.5)];
        let rects = to_face_rects(dets, 100.0, 100.0);
        assert_eq!(rects[0].x1, 0.0);
        assert_eq!(rects[0].y1, 0.0);
        assert_eq!(rects[0].x2, 100.0);
        assert_eq!(rects[0].y2, 100.0);
    }
}
