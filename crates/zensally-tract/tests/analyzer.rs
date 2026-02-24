#![cfg(all(feature = "ultraface", feature = "microsalnet"))]

use std::path::Path;
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

#[test]
fn analyzer_detects_faces_and_saliency() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");
    let result = analyzer.analyze(&image);

    // Portrait image should detect at least one face
    assert!(
        !result.faces.is_empty(),
        "portrait should have detected faces"
    );

    // Saliency should be present and non-trivial
    let saliency = result.saliency.as_ref().expect("should have saliency");
    assert_eq!(saliency.width, 128);
    assert_eq!(saliency.height, 128);

    let max_sal = saliency.data.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_sal > 0.3,
        "saliency should be non-trivial, max={max_sal:.3}"
    );
}

#[test]
fn analyzer_reusable() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = ContentAnalyzer::new().expect("failed to create analyzer");

    // Analyze twice — should produce consistent results
    let r1 = analyzer.analyze(&image);
    let r2 = analyzer.analyze(&image);

    assert_eq!(r1.faces.len(), r2.faces.len());
    assert!(r1.saliency.is_some());
    assert!(r2.saliency.is_some());
}
