//! Offline corpus adapter: decode through our JXL reader, write packed RGB8.
#![forbid(unsafe_code)]
use std::io::Write;
use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: crop-decode INPUT_JXL NEW_RGB8_OUTPUT".into());
    }
    let bytes = std::fs::read(&args[1])?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("encoded corpus source exceeds 64 MiB".into());
    }
    let mut options = JxlDecoderOptions::default();
    options.limits = JxlDecoderLimits::default()
        .with_max_pixels(32_000_000)
        .with_max_memory_bytes(2 * 1024 * 1024 * 1024);
    let image = zenjxl_decoder::decode_with(&bytes, options)?;
    if !matches!(image.channels, 2 | 4) {
        return Err("unexpected convenience decoder layout".into());
    }
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(image.width * image.height * 3)?;
    for pixel in image.data.chunks_exact(image.channels) {
        if pixel[image.channels - 1] != 255 {
            return Err("transparent corpus sources need an explicit compositing policy".into());
        }
        if image.channels == 2 {
            rgb.extend_from_slice(&[pixel[0]; 3]);
        } else {
            rgb.extend_from_slice(&pixel[..3]);
        }
    }
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[2])?
        .write_all(&rgb)?;
    // Output profile is retained, not silently relabelled by this adapter.
    println!(
        "{{\"width\":{},\"height\":{},\"profile\":{:?}}}",
        image.width,
        image.height,
        format!("{:?}", image.output_profile)
    );
    Ok(())
}
