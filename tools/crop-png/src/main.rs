//! Decode native RGB8 PNG with the owned codec for offline storage conversion.
#![forbid(unsafe_code)]
use std::io::{Read, Write};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&a.len()) {
        return Err("usage: crop-png INPUT_PNG NEW_RGB8 [MAX_PIXELS [MAX_ENCODED_BYTES]]".into());
    }
    let max_pixels = a
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(8_294_400);
    let max_bytes: usize = a
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64 * 1024 * 1024);
    if max_pixels == 0
        || max_pixels > 64 * 1024 * 1024
        || max_bytes == 0
        || max_bytes > 256 * 1024 * 1024
    {
        return Err("explicit PNG limits exceed bounded tool capacity".into());
    }
    let file = std::fs::File::open(&a[1])?;
    let len = usize::try_from(file.metadata()?.len())?;
    if len > max_bytes {
        return Err("PNG exceeds encoded-byte limit".into());
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len + 1)?;
    file.take(len as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() != len {
        return Err("PNG length changed while reading".into());
    }
    let mut c = zenpng::PngDecodeConfig::strict();
    c.max_pixels = Some(max_pixels);
    c.max_memory_bytes = Some(512 * 1024 * 1024);
    let image = zenpng::decode(&bytes, &c, &enough::Unstoppable)?;
    let p = image.pixels.as_slice();
    if p.descriptor().channels() != 3 || image.pixels.is_grayscale() {
        return Err("storage experiment requires RGB8".into());
    }
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&a[2])?;
    for y in 0..p.rows() {
        let row = p.row(y);
        if row.len() != p.width() as usize * 3 {
            return Err("expected RGB8 rows".into());
        }
        out.write_all(row)?;
    }
    println!("{{\"width\":{},\"height\":{}}}", p.width(), p.rows());
    Ok(())
}
