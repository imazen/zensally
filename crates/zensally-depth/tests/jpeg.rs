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

#[test]
fn defaults_millimeters_confidence_and_extended_xmp() {
    let png = depth_png();
    let xmp = gdepth("RangeInverse", &png)
        .replace(" GDepth:Format=\"RangeInverse\"", "")
        .replace("GDepth:Near=\"1\"", "GDepth:Near=\"1000\"")
        .replace("GDepth:Far=\"10\"", "GDepth:Far=\"10000\"")
        .replace("GDepth:Units=\"m\"", "GDepth:Units=\"mm\"")
        .replace(
            " /></rdf:RDF>",
            &format!(
                " GDepth:Confidence=\"{}\" GDepth:ConfidenceMime=\"image/png\" /></rdf:RDF>",
                base64(&png)
            ),
        );
    // Force the existing encoder's Extended XMP transport, retaining real depth.
    let xmp = xmp.replace(
        "</x:xmpmeta>",
        &format!("<!--{}--></x:xmpmeta>", "padding".repeat(10000)),
    );
    let d = decode_embedded_depth(
        &primary(&xmp),
        ImageFormat::Jpeg,
        &ResourceLimits::default(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(d.samples[0], 1.0);
    assert_eq!(d.samples[4], 10.0);
    assert_eq!(d.confidence.as_ref().unwrap()[2], 32768.0 / 65535.0);
    let q = 32768.0f64 / 65535.0;
    assert_eq!(
        d.samples[2].to_bits(),
        ((10.0 / (10.0 - q * 9.0)) as f32).to_bits()
    );
}

#[test]
fn dynamic_depth_appended_png_and_confidence() {
    let png = depth_png();
    let xmp = format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:RDF><rdf:Description xmlns:Container="http://ns.google.com/photos/dd/1.0/container/" xmlns:Item="http://ns.google.com/photos/dd/1.0/item/" xmlns:DepthMap="http://ns.google.com/photos/dd/1.0/depthmap/" DepthMap:Format="RangeLinear" DepthMap:Near="1" DepthMap:Far="10" DepthMap:Units="Meters"><Container:Directory><rdf:Seq><rdf:li><Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0"/></rdf:li><rdf:li><Container:Item Item:Mime="image/png" Item:Semantic="DepthMap" Item:Length="{n}"/></rdf:li><rdf:li><Container:Item Item:Mime="image/png" Item:Semantic="ConfidenceMap" Item:Length="{n}"/></rdf:li></rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>"#,
        n = png.len()
    );
    for (units, near, far) in [("Meters", "1", "10"), ("mm", "1000", "10000")] {
        let xmp = xmp
            .replace(
                "DepthMap:Units=\"Meters\"",
                &format!("DepthMap:Units=\"{units}\""),
            )
            .replace("DepthMap:Near=\"1\"", &format!("DepthMap:Near=\"{near}\""))
            .replace("DepthMap:Far=\"10\"", &format!("DepthMap:Far=\"{far}\""));
        let mut bytes = primary(&xmp);
        bytes.extend_from_slice(&png);
        bytes.extend_from_slice(&png);
        let d = decode_embedded_depth(&bytes, ImageFormat::Jpeg, &ResourceLimits::default())
            .unwrap()
            .unwrap();
        assert_eq!((d.width, d.height), (3, 3));
        assert_eq!(d.samples[0], 1.0);
        assert_eq!(d.samples[4], 10.0);
        assert_eq!(d.confidence.unwrap()[4], 1.0);
        let mut limits = ResourceLimits::default();
        limits.max_output_bytes = Some(9 * 4);
        assert!(decode_embedded_depth(&bytes, ImageFormat::Jpeg, &limits).is_err());
        limits.max_output_bytes = Some(9 * 4 * 2);
        assert!(decode_embedded_depth(&bytes, ImageFormat::Jpeg, &limits).is_ok());
    }
}

#[test]
fn mpf_disparity_uses_existing_encoder_and_decoder() {
    let depth = primary("");
    let image = zenjpeg::decoder::Decoder::new()
        .preserve(zenjpeg::decoder::PreserveConfig::all())
        .decode(
            &primary("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>"),
            Unstoppable,
        )
        .unwrap();
    let segments = image
        .extras()
        .unwrap()
        .to_encoder_segments()
        .add_depth_map(depth);
    let cfg = EncoderConfig::grayscale(95.0);
    let mut enc = cfg
        .request()
        .segments(segments)
        .encode_from_bytes(8, 8, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&[128; 64], Unstoppable).unwrap();
    let bytes = enc.finish().unwrap();
    let d = decode_embedded_depth(&bytes, ImageFormat::Jpeg, &ResourceLimits::default())
        .unwrap()
        .unwrap();
    assert_eq!((d.width, d.height), (8, 8));
    assert_eq!(d.direction, DepthDirection::Nearer);
    assert!(
        d.samples
            .iter()
            .all(|&v| (v - 128.0 / 255.0).abs() <= 1.0 / 255.0)
    );
}

#[test]
fn primary_exif_orientation_is_explicit_and_stored_geometry_is_preserved() {
    use zenjpeg::encoder::{Exif, Orientation};
    let xmp = gdepth("RangeLinear", &depth_png());
    let cfg = EncoderConfig::grayscale(95.0);
    let mut encoder = cfg
        .request()
        .xmp(xmp.as_bytes())
        .exif(Exif::build().orientation(Orientation::Rotate90))
        .encode_from_bytes(8, 16, PixelLayout::Gray8Srgb)
        .unwrap();
    encoder.push_packed(&[128; 128], Unstoppable).unwrap();
    let bytes = encoder.finish().unwrap();
    let depth = decode_embedded_depth(&bytes, ImageFormat::Jpeg, &ResourceLimits::default())
        .unwrap()
        .unwrap();
    assert_eq!((depth.primary_width, depth.primary_height), (8, 16));
    assert_eq!(depth.orientation, zencodec::Orientation::Rotate90);
    let cue = zensally_depth::depth_cue(depth.view().unwrap(), Default::default()).unwrap();
    assert_eq!(cue.orientation, zencodec::Orientation::Rotate90);
    let native = zenjpeg::decoder::Decoder::new()
        .auto_orient(false)
        .decode(&bytes, Unstoppable)
        .unwrap();
    let display = zenjpeg::decoder::Decoder::new()
        .decode(&bytes, Unstoppable)
        .unwrap();
    assert_eq!((native.width, native.height), (8, 16));
    assert_eq!((display.width, display.height), (16, 8));
}
