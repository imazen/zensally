use zencodec::Orientation;
use zensally::SaliencyMap;
use zensally_depth::{DepthDirection, DepthView, Options, Plane, depth_cue, fuse};
fn cue(
    data: &[f32],
    stride: usize,
    confidence: Option<&[f32]>,
    direction: DepthDirection,
) -> zensally_depth::DepthCue {
    depth_cue(
        DepthView {
            samples: Plane::new(data, 7, 7, stride).unwrap(),
            confidence: confidence.map(|c| Plane::new(c, 7, 7, stride).unwrap()),
            direction,
            orientation: Orientation::Identity,
        },
        Options {
            max_side: 7,
            radius: 2,
            separation: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
}
#[test]
fn foreground_separation_survives_strides_and_depth_direction() {
    let mut depth = vec![10.0; 49];
    for y in 2..5 {
        for x in 2..5 {
            depth[y * 7 + x] = 1.0;
        }
    }
    let a = cue(&depth, 7, None, DepthDirection::Farther);
    assert!(a.map.data[24] > 0.5);
    assert_eq!(a.map.data[0], 0.0);
    let mut strided = vec![f32::NAN; 7 * 11];
    for y in 0..7 {
        strided[y * 11..y * 11 + 7].copy_from_slice(&depth[y * 7..y * 7 + 7]);
    }
    let b = cue(&strided, 11, None, DepthDirection::Farther);
    assert_eq!(a.map.data, b.map.data);
    for d in &mut depth {
        *d = 11.0 - *d;
    }
    let c = cue(&depth, 7, None, DepthDirection::Nearer);
    assert_eq!(a.map.data, c.map.data);
}
#[test]
fn invalid_depth_preserves_rgb_bits_including_negative_zero() {
    let depth = vec![f32::NAN; 49];
    let d = cue(&depth, 7, None, DepthDirection::Farther);
    let rgb = SaliencyMap {
        data: (0..49)
            .map(|i| if i == 0 { -0.0 } else { i as f32 / 49.0 })
            .collect(),
        width: 7,
        height: 7,
    };
    let out = fuse(&rgb, &d, 0.7).unwrap();
    assert!(
        rgb.data
            .iter()
            .zip(out.data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
#[test]
fn zero_confidence_and_flat_depth_provide_no_foreground_evidence() {
    let depth = vec![5.0; 49];
    let flat = cue(&depth, 7, None, DepthDirection::Farther);
    assert!(flat.map.data.iter().all(|&v| v == 0.0));
    let confidence = vec![0.0; 49];
    let invalid = cue(&depth, 7, Some(&confidence), DepthDirection::Farther);
    assert!(invalid.confidence.iter().all(|&v| v == 0.0));
}
#[test]
fn confidence_hole_does_not_turn_into_background_depth() {
    let mut depth = vec![10.0; 49];
    depth[24] = 0.0;
    let mut confidence = vec![1.0; 49];
    confidence[24] = 0.0;
    let a = cue(&depth, 7, Some(&confidence), DepthDirection::Farther);
    depth[24] = 1000.0;
    let b = cue(&depth, 7, Some(&confidence), DepthDirection::Farther);
    assert_eq!(a.map.data, b.map.data);
    assert_eq!(a.confidence[24], 0.0);
}
