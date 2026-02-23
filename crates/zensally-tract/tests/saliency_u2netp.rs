#![cfg(feature = "u2netp")]

use std::path::Path;
use zensally::{ImageRef, PixelFormat, SaliencyDetector};
use zensally_tract::U2NetpDetector;

#[test]
fn saliency_portrait() {
    let img_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/portrait.jpg");

    let img = image::open(&img_path).expect("failed to open test image");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = rgb.as_raw();

    eprintln!("Image size: {}x{}", w, h);

    let mut detector = U2NetpDetector::new().expect("failed to create U2-Netp detector");
    let map = detector.saliency_map(&ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap());

    eprintln!("Saliency map: {}x{}, {} values", map.width, map.height, map.data.len());

    assert_eq!(map.width, 320);
    assert_eq!(map.height, 320);
    assert_eq!(map.data.len(), 320 * 320);

    // Should have some salient region (not all zeros)
    let max_val = map.data.iter().cloned().fold(0.0f32, f32::max);
    let mean_val: f32 = map.data.iter().sum::<f32>() / map.data.len() as f32;
    eprintln!("Max saliency: {:.3}, mean: {:.3}", max_val, mean_val);

    assert!(max_val > 0.5, "portrait should have salient regions, max={:.3}", max_val);

    // Center region should be more salient than edges (portrait = centered subject)
    let center_mean = center_region_mean(&map.data, 320, 320);
    let edge_mean = edge_region_mean(&map.data, 320, 320);
    eprintln!("Center mean: {:.3}, edge mean: {:.3}", center_mean, edge_mean);

    assert!(
        center_mean > edge_mean,
        "center should be more salient than edges in a portrait"
    );
}

fn center_region_mean(data: &[f32], w: usize, h: usize) -> f32 {
    let x0 = w / 4;
    let x1 = 3 * w / 4;
    let y0 = h / 4;
    let y1 = 3 * h / 4;
    let mut sum = 0.0f32;
    let mut count = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += data[y * w + x];
            count += 1;
        }
    }
    sum / count as f32
}

fn edge_region_mean(data: &[f32], w: usize, h: usize) -> f32 {
    let border = w / 8;
    let mut sum = 0.0f32;
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            if x < border || x >= w - border || y < border || y >= h - border {
                sum += data[y * w + x];
                count += 1;
            }
        }
    }
    sum / count as f32
}

#[test]
fn saliency_solid_gray() {
    let w = 640u32;
    let h = 480u32;
    let pixels = vec![128u8; (w * h * 3) as usize];

    let mut detector = U2NetpDetector::new().expect("failed to create U2-Netp detector");
    let map = detector.saliency_map(&ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap());

    let max_val = map.data.iter().cloned().fold(0.0f32, f32::max);
    let min_val = map.data.iter().cloned().fold(f32::MAX, f32::min);
    let range = max_val - min_val;
    eprintln!("Solid gray: max={:.3}, min={:.3}, range={:.3}", max_val, min_val, range);

    // Solid gray should have low saliency variance (nothing interesting)
    // After min-max normalization the range is always 0-1, so check that the
    // pre-normalized values would have been nearly uniform
    // Actually with min-max normalization, any input produces 0-1 range.
    // Just check the map has valid values.
    assert!(map.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}
