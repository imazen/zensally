//! Offline packed RGB8 to JXL storage experiment; never resizes pixels.
#![forbid(unsafe_code)]
use jxl_encoder::{Limits, LosslessConfig, LossyConfig, PixelLayout};
use std::io::Write;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: crop-compress INPUT_RGB8 WIDTH HEIGHT NEW_JXL lossless|0.5".into());
    }
    if std::path::Path::new(&a[4]).exists() {
        return Err("output exists".into());
    }
    let (w, h): (u32, u32) = (a[2].parse()?, a[3].parse()?);
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > 8_294_400 {
        return Err("RGB dimensions exceed study envelope".into());
    }
    let rgb = std::fs::read(&a[1])?;
    if rgb.len() != w as usize * h as usize * 3 {
        return Err("RGB8 length mismatch".into());
    }
    let limits = Limits::default().with_max_memory_bytes(2 * 1024 * 1024 * 1024);
    let encoded = match a[5].as_str() {
        "lossless" => LosslessConfig::new()
            .with_effort(7)
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&rgb)?,
        "0.5" => LossyConfig::new(0.5)
            .with_effort(7)
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(&rgb)?,
        _ => return Err("unsupported study mode".into()),
    };
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[4])?
        .write_all(&encoded)?;
    println!(
        "{{\"width\":{w},\"height\":{h},\"bytes\":{}}}",
        encoded.len()
    );
    Ok(())
}
