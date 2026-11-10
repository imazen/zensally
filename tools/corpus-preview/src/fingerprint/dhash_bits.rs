//! Shared bit ordering; callers own and identify decode/resample semantics.
pub(super) fn from_luma9x8<T: PartialOrd>(pixels: &[T; 72]) -> u64 {
    let mut hash = 0;
    for y in 0..8 {
        for x in 0..8 {
            if pixels[y * 9 + x] > pixels[y * 9 + x + 1] {
                hash |= 1u64 << (y * 8 + x);
            }
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::from_luma9x8;
    #[test]
    fn direction_and_bit_order() {
        let mut p = [0; 72];
        assert_eq!(from_luma9x8(&p), 0);
        p[0] = 1;
        p[7 * 9 + 7] = 1;
        assert_eq!(from_luma9x8(&p), 1 | (1 << 63));
        for y in 0..8 {
            for x in 0..9 {
                p[y * 9 + x] = (9 - x) as u8;
            }
        }
        assert_eq!(from_luma9x8(&p), u64::MAX);
    }
}
