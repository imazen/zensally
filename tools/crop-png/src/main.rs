//! Decode native RGB8 PNG with the owned codec for offline storage conversion.
#![forbid(unsafe_code)]
use std::io::Write;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: crop-png INPUT_PNG NEW_RGB8".into());
    }
    let bytes = std::fs::read(&a[1])?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("PNG exceeds study limit".into());
    }
    let mut c = zenpng::PngDecodeConfig::strict();
    c.max_pixels = Some(8_294_400);
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
