//! Owned RGB frontend to the reused exposure-normalized fingerprint kernel.
#![forbid(unsafe_code)]

use anyhow::{Result, ensure};
use std::fs::File;
use std::io::Read;
use zenpixels_convert::{PixelDescriptor, RowConverter};

#[path = "../fingerprint/dhash_bits.rs"]
mod dhash_bits;
#[path = "../fingerprint/native_luma.rs"]
mod native_luma;

fn fingerprint(bytes: &[u8], width: u32, rows: u32, stride: usize) -> Result<(u64, f32)> {
    ensure!(
        (9..=4096).contains(&width) && (8..=4096).contains(&rows),
        "bounded dimensions"
    );
    let packed = width as usize * 3;
    ensure!(
        stride >= packed && stride <= packed + 4096,
        "bounded RGB stride"
    );
    let needed = (rows as usize - 1) * stride + packed;
    ensure!(
        bytes.len() >= needed && bytes.len() <= stride * rows as usize,
        "RGB extent"
    );
    let mut converter =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR)?;
    let mut row = Vec::new();
    row.try_reserve_exact(width as usize * 12)?;
    row.resize(width as usize * 12, 0);
    let mut luma = Vec::new();
    luma.try_reserve_exact(width as usize * rows as usize)?;
    for y in 0..rows as usize {
        converter.convert_row(&bytes[y * stride..y * stride + packed], &mut row, width);
        for pixel in row.as_chunks::<12>().0 {
            let channels = [0, 1, 2].map(|i| f32::from_ne_bytes(pixel.as_chunks::<4>().0[i]));
            ensure!(
                channels.iter().all(|v| v.is_finite()),
                "nonfinite RGB sample"
            );
            luma.push(0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]);
        }
    }
    native_luma::fingerprint_luma(&mut luma, width, rows)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: corpus-fingerprint INPUT_RGB WIDTH ROWS STRIDE"
    );
    let (width, rows, stride) = (args[2].parse()?, args[3].parse()?, args[4].parse()?);
    let file = File::open(&args[1])?;
    let length = usize::try_from(file.metadata()?.len())?;
    ensure!(length <= 64 * 1024 * 1024, "64MiB encoded RGB extent");
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length + 1)?;
    file.take(length as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() == length, "input length changed");
    let (hash, peak) = fingerprint(&bytes, width, rows, stride)?;
    println!("{{\"hash\":\"{hash:016x}\",\"positive_luma_peak\":{peak}}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_and_short_final_rows_preserve_identity() {
        let (width, rows) = (37, 29);
        let packed: Vec<_> = (0..width * rows * 3)
            .map(|i| ((i * 31) % 251) as u8)
            .collect();
        let stride = width as usize * 3 + 17;
        let mut padded = vec![219; stride * rows as usize];
        for y in 0..rows as usize {
            padded[y * stride..y * stride + width as usize * 3]
                .copy_from_slice(&packed[y * width as usize * 3..(y + 1) * width as usize * 3]);
        }
        let expected = fingerprint(&packed, width, rows, width as usize * 3).unwrap();
        assert_eq!(fingerprint(&padded, width, rows, stride).unwrap(), expected);
        padded.truncate((rows as usize - 1) * stride + width as usize * 3);
        assert_eq!(fingerprint(&padded, width, rows, stride).unwrap(), expected);
    }
}
