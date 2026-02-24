#![cfg(all(feature = "ultraface", feature = "microsalnet"))]

use std::path::Path;
use zensally::crop::{
    CropConfig, CropMode, CropRect, LANDSCAPE_16_9, PORTRAIT_3_4, PORTRAIT_9_16, SQUARE,
};
use zensally::{ImageRef, PixelFormat};
use zensally_tract::ContentAnalyzer;

fn test_image() -> (Vec<u8>, u32, u32) {
    let img_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/portrait.jpg");

    let img = image::open(&img_path).expect("failed to open test image");
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    (rgb.into_raw(), w, h)
}

fn assert_crop_inside(crop: &CropRect, src_w: u32, src_h: u32) {
    assert!(
        crop.x + crop.w <= src_w,
        "crop right edge {} exceeds src width {}",
        crop.x + crop.w,
        src_w
    );
    assert!(
        crop.y + crop.h <= src_h,
        "crop bottom edge {} exceeds src height {}",
        crop.y + crop.h,
        src_h
    );
}

fn assert_nonzero(crop: &CropRect) {
    assert!(
        crop.w > 0 && crop.h > 0,
        "crop has zero dimension: {:?}",
        crop
    );
}

#[test]
fn analyzer_smoke_test() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");
    let analysis = analyzer.analyze(&image);

    // Portrait image should detect at least one face
    assert!(
        !analysis.faces.is_empty(),
        "portrait should have detected faces"
    );

    // Saliency should have non-trivial output
    let saliency = analysis.saliency.as_ref().expect("should have saliency");
    let max_sal = saliency.data.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_sal > 0.3,
        "saliency should be non-trivial, max={max_sal:.3}"
    );
}

#[test]
fn analyzer_compute_crops_batch() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");
    let analysis = analyzer.analyze(&image);

    let targets = [
        (PORTRAIT_9_16, CropMode::Minimal),
        (PORTRAIT_9_16, CropMode::Maximal),
        (SQUARE, CropMode::Minimal),
        (LANDSCAPE_16_9, CropMode::Minimal),
        (PORTRAIT_3_4, CropMode::Maximal),
    ];

    let crops = analysis.compute_crops(w, h, &targets);

    assert_eq!(crops.len(), targets.len());

    for (i, crop_opt) in crops.iter().enumerate() {
        let crop = crop_opt
            .as_ref()
            .unwrap_or_else(|| panic!("target {i} returned None"));
        assert_nonzero(crop);
        assert_crop_inside(crop, w, h);
    }
}

#[test]
fn analyzer_compute_single_crop() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");
    let analysis = analyzer.analyze(&image);

    let config = CropConfig {
        target_aspect: SQUARE,
        mode: CropMode::Minimal,
        face_vertical_position: 0.35,
        min_face_visibility: 0.8,
        zoom_padding: 0.3,
    };

    let crop = analysis
        .compute_crop(w, h, &config)
        .expect("single crop should succeed");

    assert_nonzero(&crop);
    assert_crop_inside(&crop, w, h);

    // Square crop should have approximately equal dimensions
    let ratio = crop.w as f64 / crop.h as f64;
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "square crop should be square, ratio={ratio:.4}"
    );
}

#[test]
fn analyzer_batch_matches_individual() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");
    let analysis = analyzer.analyze(&image);

    let targets = [
        (PORTRAIT_9_16, CropMode::Minimal),
        (SQUARE, CropMode::Maximal),
        (LANDSCAPE_16_9, CropMode::Minimal),
    ];

    let batch = analysis.compute_crops(w, h, &targets);

    // Each batch result should match a single-crop call with the same default config
    for (i, &(ratio, mode)) in targets.iter().enumerate() {
        let config = CropConfig {
            target_aspect: ratio,
            mode,
            ..CropConfig::default()
        };
        let single = analysis.compute_crop(w, h, &config);
        assert_eq!(
            batch[i], single,
            "batch[{i}] should match individual compute_crop"
        );
    }
}
