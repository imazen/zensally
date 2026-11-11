#![cfg(all(feature = "yunet", feature = "microsalnet", feature = "selfie_seg"))]

use std::path::Path;
use zensally::sat_crop::SatCropMode;
use zensally::{ImageRef, PixelFormat};
use zensally_tract::{CropAnalyzerV2, YuNetDetector};

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
fn yunet_landmarks_decode_on_portrait() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut detector = YuNetDetector::new().expect("failed to create YuNet detector");
    let faces = detector.detect_with_landmarks(&image);

    assert!(!faces.is_empty(), "portrait should have detected faces");
    let f = &faces[0];
    let kps = f.landmarks.expect("YuNet faces carry 5-point landmarks");
    eprintln!("face {:?} kps {kps:?}", f.rect);
    // Landmark percentages should sit inside the detected face box.
    for p in kps {
        assert!(
            p[0] > f.rect.x1 - 10.0 && p[0] < f.rect.x2 + 10.0,
            "landmark x {} outside face {:?}",
            p[0],
            f.rect
        );
        assert!(
            p[1] > f.rect.y1 - 10.0 && p[1] < f.rect.y2 + 10.0,
            "landmark y {} outside face {:?}",
            p[1],
            f.rect
        );
    }
}

#[test]
fn crop_analyzer_v2_portrait() {
    let (pixels, w, h) = test_image();
    let image = ImageRef::new(&pixels, w, h, PixelFormat::Rgb).unwrap();

    let mut analyzer = CropAnalyzerV2::new().expect("failed to create CropAnalyzerV2");

    let requests: Vec<((u32, u32), SatCropMode)> = [(9, 16), (3, 4), (1, 1), (16, 9)]
        .into_iter()
        .flat_map(|a| [SatCropMode::Minimal, SatCropMode::Maximal].map(|m| (a, m)))
        .collect();
    let crops = analyzer.compute_crops(&image, &requests, &[]);
    assert_eq!(crops.len(), requests.len());

    for (i, c) in crops.iter().enumerate() {
        let r = c.unwrap_or_else(|| panic!("request {i} produced no crop"));
        assert!(r.w > 0 && r.h > 0, "request {i} empty rect {r:?}");
        assert!(
            r.x + r.w <= w && r.y + r.h <= h,
            "request {i} rect {r:?} out of {w}x{h} bounds"
        );
        // Aspect must match the request within integer snapping (~2%).
        let (aw, ah) = requests[i].0;
        let got = r.w as f64 / r.h as f64;
        let want = aw as f64 / ah as f64;
        assert!(
            (got / want - 1.0).abs() < 0.02,
            "request {i} aspect {got:.3} vs {want:.3} ({r:?})"
        );
    }

    // Single-shot convenience path must agree with the batch result.
    let single = analyzer.compute_crop(&image, (3, 4), SatCropMode::Maximal, &[]);
    let idx = requests
        .iter()
        .position(|&r| r == ((3, 4), SatCropMode::Maximal))
        .unwrap();
    assert_eq!(single, crops[idx]);
}
