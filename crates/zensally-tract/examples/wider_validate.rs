#![forbid(unsafe_code)]

//! WIDER FACE validation benchmark with face-size filtering.
//!
//! Run: `cargo run --package zensally-tract --example wider_validate --release`
//! With blazeface320: `cargo run --package zensally-tract --example wider_validate --release --features blazeface320`
//!
//! Requires WIDER FACE dataset downloaded to `data/wider_face/`.
//! See `scripts/download_wider_face.sh`.

use std::path::{Path, PathBuf};
use std::time::Instant;
use zensally::{FaceDetector, FaceRect, ImageRef, PixelFormat};

/// A ground-truth face annotation.
struct GtFace {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    invalid: bool,
}

/// A single annotated image with pre-loaded dimensions.
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
                faces.push(GtFace { x, y, w, h, invalid });
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

    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Match detections to size-filtered ground truth.
/// Only GT faces with min(w,h) / min(img_w,img_h) >= min_face_frac are considered.
fn match_detections(
    detections: &[FaceRect],
    gt_faces: &[GtFace],
    img_w: f32,
    img_h: f32,
    iou_threshold: f32,
    min_face_frac: f32,
) -> (Vec<(f32, bool)>, usize) {
    let img_short = img_w.min(img_h);
    let valid_gt: Vec<&GtFace> = gt_faces
        .iter()
        .filter(|f| !f.invalid && f.w.min(f.h) / img_short >= min_face_frac)
        .collect();
    let n_gt = valid_gt.len();
    let mut matched = vec![false; n_gt];
    let mut results = Vec::new();

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

/// Compute AP using 11-point interpolation (PASCAL VOC style).
fn compute_ap(scored_results: &mut [(f32, bool)], total_gt: usize) -> f32 {
    if total_gt == 0 {
        return 0.0;
    }

    scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut precisions = Vec::with_capacity(scored_results.len());
    let mut recalls = Vec::with_capacity(scored_results.len());

    for &(_, is_tp) in scored_results.iter() {
        if is_tp { tp += 1; } else { fp += 1; }
        precisions.push(tp as f32 / (tp + fp) as f32);
        recalls.push(tp as f32 / total_gt as f32);
    }

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

/// Run validation for one detector at multiple min-face-fraction thresholds.
/// Each threshold is min(face_w, face_h) / min(img_w, img_h).
fn run_validation(
    name: &str,
    detector: &mut dyn FaceDetector,
    annotations: &[AnnotatedImage],
    img_dir: &Path,
    frac_thresholds: &[f32],
) {
    println!("\n--- {name} ---");

    // First pass: run detector on all images, collect detections + timings.
    let mut per_image: Vec<(Vec<FaceRect>, u32, u32)> = Vec::with_capacity(annotations.len());
    let mut total_inference_ms = 0.0f64;
    let mut images_processed = 0usize;
    let mut images_skipped = 0usize;

    let start = Instant::now();

    for (i, ann) in annotations.iter().enumerate() {
        let img_path = img_dir.join(&ann.path);
        if !img_path.exists() {
            images_skipped += 1;
            per_image.push((Vec::new(), 0, 0));
            continue;
        }

        let img = match image::open(&img_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  WARN: failed to open {}: {e}", ann.path);
                images_skipped += 1;
                per_image.push((Vec::new(), 0, 0));
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

        per_image.push((detections, w, h));
        images_processed += 1;

        if (i + 1) % 500 == 0 || i + 1 == annotations.len() {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = images_processed as f64 / elapsed;
            eprint!(
                "\r  [{}/{}] {:.0} img/s, avg {:.1}ms/img",
                i + 1,
                annotations.len(),
                fps,
                total_inference_ms / images_processed as f64,
            );
        }
    }
    eprintln!();

    let total_elapsed = start.elapsed().as_secs_f64();
    let total_detections: usize = per_image.iter().map(|(d, _, _)| d.len()).sum();

    println!("  Images: {images_processed} processed, {images_skipped} skipped");
    println!("  Detections: {total_detections}");
    println!(
        "  Avg inference: {:.1}ms/image",
        total_inference_ms / images_processed as f64
    );
    println!(
        "  Total: {:.1}s ({:.0} img/s incl. I/O)",
        total_elapsed,
        images_processed as f64 / total_elapsed,
    );

    // Second pass: evaluate at each face-fraction threshold.
    println!();
    println!("  {:>10}  {:>8}  {:>8}  {:>6}  {:>6}  {:>7}", "min_frac", "gt_faces", "matched", "prec", "recall", "AP");
    println!("  {:->10}  {:->8}  {:->8}  {:->6}  {:->6}  {:->7}", "", "", "", "", "", "");

    for &min_frac in frac_thresholds {
        let mut all_results: Vec<(f32, bool)> = Vec::new();
        let mut total_gt = 0usize;
        let mut total_tp = 0usize;

        for (ann, (detections, w, h)) in annotations.iter().zip(per_image.iter()) {
            if *w == 0 {
                continue;
            }
            let (mut results, n_gt) =
                match_detections(detections, &ann.faces, *w as f32, *h as f32, 0.5, min_frac);
            total_gt += n_gt;
            total_tp += results.iter().filter(|(_, tp)| *tp).count();
            all_results.append(&mut results);
        }

        let ap = compute_ap(&mut all_results, total_gt);
        let precision = if total_detections > 0 {
            total_tp as f32 / total_detections as f32
        } else {
            0.0
        };
        let recall = if total_gt > 0 {
            total_tp as f32 / total_gt as f32
        } else {
            0.0
        };

        let label = if min_frac <= 0.0 {
            "all".to_string()
        } else {
            format!("{:.0}%", min_frac * 100.0)
        };

        println!(
            "  {:>10}  {:>8}  {:>8}  {:>5.1}%  {:>5.1}%  {:>6.1}%",
            label,
            total_gt,
            total_tp,
            precision * 100.0,
            recall * 100.0,
            ap * 100.0,
        );
    }
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

    if !ann_path.exists() || !img_dir.exists() {
        eprintln!("WIDER FACE not found at {}", data_dir.display());
        eprintln!("Run: bash scripts/download_wider_face.sh");
        std::process::exit(1);
    }

    println!("Parsing annotations...");
    let annotations = parse_annotations(&ann_path);
    println!("  {} images", annotations.len());

    let total_faces: usize = annotations.iter().map(|a| a.faces.len()).sum();
    let valid_faces: usize = annotations
        .iter()
        .flat_map(|a| a.faces.iter())
        .filter(|f| !f.invalid)
        .count();
    println!("  {total_faces} total GT faces ({valid_faces} valid)");

    // Face size distribution as fraction of image short edge.
    // Need image dimensions, so load them.
    println!("  Loading image dimensions...");
    let img_dims: Vec<(u32, u32)> = annotations
        .iter()
        .map(|ann| {
            let img_path = img_dir.join(&ann.path);
            image::image_dimensions(&img_path).unwrap_or((0, 0))
        })
        .collect();

    let mut fracs: Vec<f32> = annotations
        .iter()
        .zip(img_dims.iter())
        .flat_map(|(a, &(iw, ih))| {
            let short = (iw.min(ih)) as f32;
            a.faces
                .iter()
                .filter(|f| !f.invalid)
                .map(move |f| f.w.min(f.h) / short)
        })
        .collect();
    fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !fracs.is_empty() {
        let p = |pct: usize| fracs[pct * fracs.len() / 100] * 100.0;
        println!(
            "  Face/image ratio: p10={:.1}% p25={:.1}% p50={:.1}% p75={:.1}% p90={:.1}%",
            p(10), p(25), p(50), p(75), p(90)
        );
    }

    // Fraction thresholds: face min dim / image short edge.
    // For smart crop, faces >= ~5-10% of image are what matter.
    let frac_thresholds: &[f32] = &[0.0, 0.05, 0.10, 0.15, 0.20, 0.30];

    // --- MediaPipe 128 (speed baseline, feature-gated) ---
    #[cfg(feature = "mediapipe")]
    {
        let config = zensally_tract::MediaPipeBlazeFaceConfig {
            raw_score_threshold: 0.0,
            min_confidence: 0.5,
            nms_iou_threshold: 0.3,
        };
        let mut det = zensally_tract::MediaPipeBlazeFaceDetector::with_config(config)
            .expect("failed to create MediaPipe detector");
        run_validation(
            "MediaPipe 128 (conf>=0.5)",
            &mut det,
            &annotations,
            &img_dir,
            frac_thresholds,
        );
    }

    // --- BlazeFace-320 (feature-gated) ---
    #[cfg(feature = "blazeface320")]
    {
        println!("\nLoading BlazeFace-320...");
        let mut det =
            zensally_tract::BlazeFaceDetector::new().expect("failed to create BlazeFace-320 detector");
        run_validation("BlazeFace 320", &mut det, &annotations, &img_dir, frac_thresholds);
    }

    // --- YuNet 640 (feature-gated) ---
    #[cfg(feature = "yunet")]
    {
        println!("\nLoading YuNet 640...");
        let mut det =
            zensally_tract::YuNetDetector::new().expect("failed to create YuNet detector");
        run_validation("YuNet 640", &mut det, &annotations, &img_dir, frac_thresholds);

        // Also test with a lower threshold for more recall
        let config = zensally_tract::YuNetConfig {
            score_threshold: 0.3,
            nms_iou_threshold: 0.3,
        };
        let mut det =
            zensally_tract::YuNetDetector::with_config(config).expect("failed to create YuNet detector");
        run_validation("YuNet 640 (conf>=0.3)", &mut det, &annotations, &img_dir, frac_thresholds);
    }

    // --- UltraFace RFB-320 (feature-gated) ---
    #[cfg(feature = "ultraface")]
    {
        println!("\nLoading UltraFace RFB-320...");
        let mut det =
            zensally_tract::UltraFaceDetector::new().expect("failed to create UltraFace detector");
        run_validation("UltraFace 320 (conf>=0.7)", &mut det, &annotations, &img_dir, frac_thresholds);

        // Also test with lower threshold
        let config = zensally_tract::UltraFaceConfig {
            score_threshold: 0.5,
            nms_iou_threshold: 0.3,
        };
        let mut det =
            zensally_tract::UltraFaceDetector::with_config(config).expect("failed to create UltraFace detector");
        run_validation("UltraFace 320 (conf>=0.5)", &mut det, &annotations, &img_dir, frac_thresholds);
    }
}
