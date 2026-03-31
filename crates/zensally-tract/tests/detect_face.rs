#![cfg(feature = "blazeface320")]

use std::path::Path;
use zensally::{FaceDetector, ImageRef, PixelFormat};
use zensally_tract::BlazeFaceDetector;

#[test]
fn detect_portrait() {
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

    let image_ref = ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut detector = BlazeFaceDetector::new().expect("failed to create detector");
    let faces = detector.detect(&image_ref);

    eprintln!("Detected {} faces:", faces.len());
    for (i, face) in faces.iter().enumerate() {
        eprintln!(
            "  Face {}: ({:.1}%, {:.1}%) - ({:.1}%, {:.1}%) confidence={:.3}",
            i, face.x1, face.y1, face.x2, face.y2, face.confidence
        );
    }

    assert!(
        !faces.is_empty(),
        "should detect at least one face in portrait"
    );
    assert!(
        faces[0].confidence > 0.9,
        "first face should have high confidence, got {}",
        faces[0].confidence
    );

    // Face should be roughly centered in a 1024x1024 portrait
    let face = &faces[0];
    let cx = (face.x1 + face.x2) / 2.0;
    let cy = (face.y1 + face.y2) / 2.0;
    assert!(
        cx > 20.0 && cx < 80.0,
        "face center x should be roughly centered, got {:.1}%",
        cx
    );
    assert!(
        cy > 10.0 && cy < 70.0,
        "face center y should be in upper half, got {:.1}%",
        cy
    );
}

#[test]
fn detect_no_faces_in_solid_color() {
    let w = 640u32;
    let h = 480u32;
    let pixels = vec![128u8; (w * h * 3) as usize];

    let image_ref = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut detector = BlazeFaceDetector::new().expect("failed to create detector");
    let faces = detector.detect(&image_ref);

    eprintln!("Solid gray: {} faces detected", faces.len());
    assert!(
        faces.is_empty(),
        "should not detect faces in solid gray image"
    );
}
