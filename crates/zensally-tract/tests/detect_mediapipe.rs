#![cfg(feature = "mediapipe")]

use std::path::Path;
use zensally::{FaceDetector, ImageRef, PixelFormat};
use zensally_tract::MediaPipeBlazeFaceDetector;

#[test]
fn detect_portrait_mediapipe() {
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

    let mut detector = MediaPipeBlazeFaceDetector::new().expect("failed to create detector");
    let faces = detector.detect(&ImageRef::new(pixels, w, h, PixelFormat::Rgb).unwrap());

    eprintln!("MediaPipe detected {} faces:", faces.len());
    for (i, face) in faces.iter().enumerate() {
        eprintln!(
            "  Face {}: ({:.1}%, {:.1}%) - ({:.1}%, {:.1}%) confidence={:.3}",
            i, face.x1, face.y1, face.x2, face.y2, face.confidence
        );
    }

    assert!(!faces.is_empty(), "should detect at least one face in portrait");
    assert!(
        faces[0].confidence > 0.7,
        "first face should have high confidence, got {}",
        faces[0].confidence
    );
}

#[test]
fn mediapipe_no_faces_solid() {
    let w = 640u32;
    let h = 480u32;
    let pixels = vec![128u8; (w * h * 3) as usize];

    let mut detector = MediaPipeBlazeFaceDetector::new().expect("failed to create detector");
    let faces = detector.detect(&ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap());

    eprintln!("Solid gray: {} faces detected", faces.len());
    assert!(faces.is_empty(), "should not detect faces in solid gray image");
}
