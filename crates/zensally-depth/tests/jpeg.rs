#![cfg(feature = "jpeg")]
use enough::Unstoppable;
use zencodec::{ImageFormat, ResourceLimits};
use zenjpeg::encoder::{EncoderConfig, PixelLayout};
use zensally_depth::{DepthDirection, decode_embedded_depth};

fn base64(data: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for bytes in data.chunks(3) {
        let value = (u32::from(bytes[0]) << 16)
            | (u32::from(*bytes.get(1).unwrap_or(&0)) << 8)
            | u32::from(*bytes.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i > bytes.len() {
                '='
            } else {
                CHARS[((value >> shift) & 63) as usize] as char
            });
        }
    }
    out
}
fn primary(xmp: &str) -> Vec<u8> {
    let cfg = EncoderConfig::grayscale(95.0);
    let mut enc = cfg
        .request()
        .xmp(xmp.as_bytes())
        .encode_from_bytes(8, 8, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&[128; 64], Unstoppable).unwrap();
    enc.finish().unwrap()
}
fn depth_png() -> Vec<u8> {
    let p: Vec<rgb::Gray<u16>> = [0, 16384, 32768, 49152, 65535, 0, 16384, 32768, 65535]
        .map(rgb::Gray::new)
        .to_vec();
    zenpng::encode_gray16(
        imgref::Img::new(p, 3, 3).as_ref(),
        None,
        &zenpng::EncodeConfig::default(),
        &Unstoppable,
        &Unstoppable,
    )
    .unwrap()
}
fn gdepth(format: &str, data: &[u8]) -> String {
    format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:GDepth="http://ns.google.com/photos/1.0/depthmap/" GDepth:Format="{format}" GDepth:Near="1" GDepth:Far="10" GDepth:Mime="image/png" GDepth:Units="m" GDepth:Data="{}" /></rdf:RDF></x:xmpmeta>"#,
        base64(data)
    )
}
#[test]
fn gdepth_preserves_integer_codes_and_google_inverse_direction() {
    let png = depth_png();
    for format in ["RangeLinear", "RangeInverse"] {
        let bytes = primary(&gdepth(format, &png));
        let d = decode_embedded_depth(&bytes, ImageFormat::Jpeg, &ResourceLimits::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            (d.width, d.height, d.primary_width, d.primary_height),
            (3, 3, 8, 8)
        );
        assert_eq!(d.direction, DepthDirection::Farther);
        assert_eq!(d.samples[0], 1.0);
        assert_eq!(d.samples[4], 10.0);
        let q = 32768.0f64 / 65535.0;
        let expected = if format == "RangeInverse" {
            (10.0 / (10.0 - q * 9.0)) as f32
        } else {
            1.0 + (32768.0f32 / 65535.0) * 9.0
        };
        assert_eq!(d.samples[2].to_bits(), expected.to_bits());
    }
}
#[test]
fn absent_and_malformed_declared_depth_are_distinct() {
    let no_depth = primary("");
    assert!(
        decode_embedded_depth(&no_depth, ImageFormat::Jpeg, &ResourceLimits::default())
            .unwrap()
            .is_none()
    );
    let broken = primary(&gdepth("RangeLinear", b"not a PNG"));
    assert!(decode_embedded_depth(&broken, ImageFormat::Jpeg, &ResourceLimits::default()).is_err());
    let mut limits = ResourceLimits::default();
    limits.max_input_bytes = Some(1);
    assert!(decode_embedded_depth(&no_depth, ImageFormat::Jpeg, &limits).is_err());
}
