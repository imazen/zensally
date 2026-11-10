use codec_corpus::Corpus;
use sha2::{Digest, Sha256};
use zencodec::{ImageFormat, ResourceLimits};
use zensally_depth::decode_embedded_depth;

#[test]
fn real_portrait_auxiliary_depth() {
    // Pinned external corpus, cached through the shared corpus tooling. Missing
    // network/data is a hard failure; callers select this target explicitly.
    let corpus = Corpus::new().unwrap();
    let root = corpus.tar_url("openize-heic-1640f6ed", "https://github.com/openize-com/openize-heic-net/archive/1640f6ed26dd4038b86ed9d548a7b34c2cdd7ba1.tar.gz").unwrap();
    let path = root.join("Openize.Heic.Tests/TestsData/samples/iphone_portrait_photo.heic");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        sha256(&bytes),
        "c782e03d1ef2a709f6a84ff1b1612ef2dfa909e4872958d3975e15c79ab523c3"
    );
    let native = heic::DecoderConfig::new()
        .decode_depth_with_limits(&bytes, &heic::Limits::default(), &enough::Unstoppable)
        .unwrap();
    // Independent libheif 1.21.2 + libde265 oracle; reproduce with
    // tools/heic_depth_oracle.py. Hash includes every u16 code, row packed LE.
    assert_eq!(
        (native.width, native.height, native.bit_depth),
        (576, 768, 8)
    );
    assert_eq!(
        native.depth_info.representation_type,
        heic::DepthRepresentationType::UniformDisparity
    );
    assert_eq!(native.depth_info.d_min, Some(0.53662109375));
    assert_eq!(native.depth_info.d_max, Some(3.56640625));
    let raw: Vec<u8> = native.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        sha256(&raw),
        "4fc1e83b7bd9328f894b808c718ea94f24f2713824f4f215fd8a2de589622abc"
    );
    let mut tight = heic::Limits::default();
    tight.max_pixels = Some(1);
    assert!(
        heic::DecoderConfig::new()
            .decode_depth_with_limits(&bytes, &tight, &enough::Unstoppable)
            .is_err()
    );
    let depth = decode_embedded_depth(&bytes, ImageFormat::Heic, &ResourceLimits::default())
        .unwrap()
        .expect("portrait fixture must contain depth");
    assert!(depth.width > 1 && depth.height > 1);
    assert_eq!(depth.samples.len(), depth.width * depth.height);
    assert!(depth.samples.iter().all(|v| v.is_finite()));
    let first = depth.samples[0];
    assert!(depth.samples.iter().any(|&v| v != first));
    assert_eq!(
        (
            depth.width,
            depth.height,
            depth.primary_width,
            depth.primary_height
        ),
        (576, 768, 3024, 4032)
    );
    for (&sample, &code) in depth.samples.iter().zip(&native.data) {
        assert_eq!(sample.to_bits(), (code as f32 / 255.0).to_bits());
    }
    println!(
        "depth={}x{} primary={}x{} direction={:?}",
        depth.width, depth.height, depth.primary_width, depth.primary_height, depth.direction
    );
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
