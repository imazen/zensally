#![forbid(unsafe_code)]

//! WIDER FACE validation benchmark.
//!
//! Run: `cargo run --package zenfaces-tract --example wider_validate --release`
//!
//! Requires WIDER FACE dataset downloaded to `data/wider_face/`.
//! See `scripts/download_wider_face.sh`.

use std::path::{Path, PathBuf};
use std::time::Instant;
use zenfaces::{FaceDetector, FaceRect, ImageRef, PixelFormat};
use zenfaces_tract::MediaPipeBlazeFaceDetector;

/// A ground-truth face annotation.
struct GtFace {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    invalid: bool,
}

/// A single annotated image.
struct AnnotatedImage {
    path: String,
    faces: Vec<GtFace>,
}

/// Parse wider_face_val_bbx_gt.txt annotation file.
fn parse_annotations(path: &Path) -> Vec<AnnotatedImage> {
    let content = std::fs::read_to_string(path).expect("failed to read annotation file");
    let mut lines = content.lines();
    let mut images = Vec::new();

    while let Some(image_path) = lines.next() {
        let image_path = image_path.trim();
        if image_path.is_empty() {
            continue;
        }

        let count_line = lines.next().expect("expected face count");
        let count: usize = count_line.trim().parse().expect("invalid face count");

        let mut faces = Vec::with_capacity(count);
        let n_lines = if count == 0 { 1 } else { count };

        for _ in 0..n_lines {
            let line = lines.next().expect("expected face annotation");
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 10 {
                continue;
            }

            let x: f32 = parts[0].parse().unwrap_or(0.0);
            let y: f32 = parts[1].parse().unwrap_or(0.0);
            let w: f32 = parts[2].parse().unwrap_or(0.0);
            let h: f32 = parts[3].parse().unwrap_or(0.0);
            let invalid = parts[7] == "1";

            if count > 0 && w > 0.0 && h > 0.0 {
                faces.push(GtFace {
                    x,
                    y,
                    w,
                    h,
                    invalid,
                });
            }
        }

        images.push(AnnotatedImage {
            path: image_path.to_string(),
            faces,
        });
    }

    images
}

/// Compute IoU between a detection (percentage coords) and ground truth (pixel coords).
fn iou_pct_px(det: &FaceRect, gt: &GtFace, img_w: f32, img_h: f32) -> f32 {
    // Convert detection from percentage to pixel coordinates
    let dx1 = det.x1 / 100.0 * img_w;
    let dy1 = det.y1 / 100.0 * img_h;
    let dx2 = det.x2 / 100.0 * img_w;
    let dy2 = det.y2 / 100.0 * img_h;

    let gx1 = gt.x;
    let gy1 = gt.y;
    let gx2 = gt.x + gt.w;
    let gy2 = gt.y + gt.h;

    let inter_x1 = dx1.max(gx1);
    let inter_y1 = dy1.max(gy1);
    let inter_x2 = dx2.min(gx2);
    let inter_y2 = dy2.min(gy2);

    let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let area_det = (dx2 - dx1) * (dy2 - dy1);
    let area_gt = gt.w * gt.h;
    let union = area_det + area_gt - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Match detections to ground truth, return (true_positives, false_positives, n_gt).
fn match_detections(
    detections: &[FaceRect],
    gt_faces: &[GtFace],
    img_w: f32,
    img_h: f32,
    iou_threshold: f32,
) -> (Vec<(f32, bool)>, usize) {
    // scored_results: (confidence, is_tp)
    let mut results = Vec::new();
    let valid_gt: Vec<&GtFace> = gt_faces.iter().filter(|f| !f.invalid).collect();
    let n_gt = valid_gt.len();
    let mut matched = vec![false; valid_gt.len()];

    // Detections should already be sorted by confidence (highest first)
    for det in detections {
        let mut best_iou = 0.0f32;
        let mut best_idx = None;

        for (i, gt) in valid_gt.iter().enumerate() {
            if matched[i] {
                continue;
            }
            let iou = iou_pct_px(det, gt, img_w, img_h);
            if iou > best_iou {
                best_iou = iou;
                best_idx = Some(i);
            }
        }

        if best_iou >= iou_threshold
            && let Some(idx) = best_idx
        {
            matched[idx] = true;
            results.push((det.confidence, true));
            continue;
        }
        results.push((det.confidence, false));
    }

    (results, n_gt)
}

/// Compute Average Precision from scored results using all-points interpolation.
fn compute_ap(scored_results: &mut [(f32, bool)], total_gt: usize) -> f32 {
    if total_gt == 0 {
        return 0.0;
    }

    // Sort by confidence descending
    scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp_cumsum = 0usize;
    let mut fp_cumsum = 0usize;
    let mut precisions = Vec::with_capacity(scored_results.len());
    let mut recalls = Vec::with_capacity(scored_results.len());

    for &(_, is_tp) in scored_results.iter() {
        if is_tp {
            tp_cumsum += 1;
        } else {
            fp_cumsum += 1;
        }
        let precision = tp_cumsum as f32 / (tp_cumsum + fp_cumsum) as f32;
        let recall = tp_cumsum as f32 / total_gt as f32;
        precisions.push(precision);
        recalls.push(recall);
    }

    // 11-point interpolation (PASCAL VOC style)
    let mut ap = 0.0;
    for t in 0..=10 {
        let threshold = t as f32 / 10.0;
        let max_prec = precisions
            .iter()
            .zip(recalls.iter())
            .filter(|&(_, &r)| r >= threshold)
            .map(|(&p, _)| p)
            .fold(0.0f32, f32::max);
        ap += max_prec;
    }
    ap / 11.0
}

fn main() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let data_dir = workspace_root.join("data/wider_face");
    let ann_path = data_dir.join("wider_face_val_bbx_gt.txt");
    let img_dir = data_dir.join("WIDER_val/images");

    if !ann_path.exists() {
        eprintln!("Annotation file not found: {}", ann_path.display());
        eprintln!("Run: bash scripts/download_wider_face.sh");
        std::process::exit(1);
    }
    if !img_dir.exists() {
        eprintln!("Image directory not found: {}", img_dir.display());
        eprintln!("Run: bash scripts/download_wider_face.sh");
        std::process::exit(1);
    }

    println!("Parsing annotations...");
    let annotations = parse_annotations(&ann_path);
    println!("  {} images annotated", annotations.len());

    let total_gt_faces: usize = annotations.iter().map(|a| a.faces.len()).sum();
    println!("  {} total ground truth faces", total_gt_faces);

    println!("\nLoading MediaPipe BlazeFace 128x128...");
    let load_start = Instant::now();
    let mut detector = MediaPipeBlazeFaceDetector::new().expect("failed to create detector");
    println!("  Model loaded in {:.1}ms", load_start.elapsed().as_secs_f64() * 1000.0);

    println!("\nRunning validation...");
    let mut all_results: Vec<(f32, bool)> = Vec::new();
    let mut total_gt = 0usize;
    let mut total_detections = 0usize;
    let mut total_inference_ms = 0.0f64;
    let mut images_processed = 0usize;
    let mut images_skipped = 0usize;

    let start = Instant::now();

    for (i, ann) in annotations.iter().enumerate() {
        let img_path = img_dir.join(&ann.path);
        if !img_path.exists() {
            images_skipped += 1;
            continue;
        }

        let img = match image::open(&img_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  WARN: failed to open {}: {}", ann.path, e);
                images_skipped += 1;
                continue;
            }
        };

        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let pixels = rgb.as_raw();
        let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

        let t0 = Instant::now();
        let detections = detector.detect(&image_ref);
        total_inference_ms += t0.elapsed().as_secs_f64() * 1000.0;

        total_detections += detections.len();

        let (mut results, n_gt) =
            match_detections(&detections, &ann.faces, w as f32, h as f32, 0.5);
        all_results.append(&mut results);
        total_gt += n_gt;

        images_processed += 1;
        if (i + 1) % 500 == 0 || i + 1 == annotations.len() {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = images_processed as f64 / elapsed;
            print!(
                "\r  [{}/{}] {:.0} img/s, avg {:.1}ms/img",
                i + 1,
                annotations.len(),
                fps,
                total_inference_ms / images_processed as f64
            );
        }
    }
    println!();

    let total_elapsed = start.elapsed().as_secs_f64();

    // Compute AP
    let ap = compute_ap(&mut all_results, total_gt);

    println!("\n=== WIDER FACE Validation Results ===");
    println!("Images:       {} processed, {} skipped", images_processed, images_skipped);
    println!("Ground truth: {} faces (valid, non-invalid)", total_gt);
    println!("Detections:   {}", total_detections);
    println!();
    println!("AP (IoU=0.5): {:.1}%", ap * 100.0);
    println!();
    println!("Avg inference:  {:.1}ms/image", total_inference_ms / images_processed as f64);
    println!("Total time:     {:.1}s ({:.0} img/s including I/O)",
        total_elapsed, images_processed as f64 / total_elapsed);
}
