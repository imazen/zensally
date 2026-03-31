#![cfg(feature = "selfie_seg")]

use std::path::Path;
use zensally::{ImageRef, PixelFormat, SaliencyDetector};
use zensally_tract::SelfieSeg;

#[test]
fn selfie_seg_portrait() {
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

    let mut detector = SelfieSeg::new().expect("failed to create detector");
    let map = detector.saliency_map(&ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap());

    assert_eq!(map.width, 256);
    assert_eq!(map.height, 256);
    assert_eq!(map.data.len(), 256 * 256);

    let max_val = map.data.iter().cloned().fold(0.0f32, f32::max);
    let mean_val: f32 = map.data.iter().sum::<f32>() / map.data.len() as f32;
    eprintln!("Max saliency: {:.3}, mean: {:.3}", max_val, mean_val);

    // Portrait should have strong person segmentation
    assert!(
        max_val > 0.5,
        "portrait should detect person, max={:.3}",
        max_val
    );

    // Center region should be more salient than edges (centered subject)
    let center_mean = center_region_mean(&map.data, 256, 256);
    let edge_mean = edge_region_mean(&map.data, 256, 256);
    eprintln!(
        "Center mean: {:.3}, edge mean: {:.3}",
        center_mean, edge_mean
    );

    assert!(
        center_mean > edge_mean,
        "center should be more salient than edges in a portrait"
    );
}

#[test]
fn selfie_seg_no_person() {
    // Solid gray should have no person
    let w = 640u32;
    let h = 480u32;
    let pixels = vec![128u8; (w * h * 3) as usize];

    let mut detector = SelfieSeg::new().expect("failed to create detector");
    let map = detector.saliency_map(&ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap());

    let max_val = map.data.iter().cloned().fold(0.0f32, f32::max);
    eprintln!("Solid gray max saliency: {:.3}", max_val);

    // Should have very low saliency (no person)
    assert!(
        max_val < 0.1,
        "solid gray should have no person detection, max={:.3}",
        max_val
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
