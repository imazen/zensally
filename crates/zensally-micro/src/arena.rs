//! Pre-allocated ping-pong scratch for the micro-graph forward pass.
//!
//! One `ScratchArena` carries every intermediate activation the static graph
//! needs: two ping-pong buffers (`A`/`B`) sized for the largest feature map
//! (the stride-2 stem output), plus the two FPN skip buffers (`SkipS8`,
//! `SkipS16`). A warm forward pass performs zero heap allocations — callers
//! keep the arena across frames.
//!
//! Two storage modes:
//! - `F32`: working slices are `&mut [f32]` directly.
//! - `F16`: activations are stored as packed IEEE-754 binary16 (`u16`) and
//!   widened to `f32` into row/tile scratch during kernels — halves the
//!   resident footprint. Conversion is pure safe Rust (see
//!   [`f16_to_f32`]/[`f32_to_f16`]).

use alloc::vec::Vec;

/// Which arena buffer a graph stage reads or writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BufId {
    /// Ping-pong buffer A (stride-2 stem output size).
    A,
    /// Ping-pong buffer B (same size as A).
    B,
    /// FPN skip buffer for the stride-8 feature map.
    SkipS8,
    /// FPN skip buffer for the stride-16 feature map.
    SkipS16,
}

/// Arena element storage: plain `f32` or packed binary16.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageMode {
    F32,
    F16,
}

enum Store {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

impl Store {
    fn new(mode: StorageMode, elems: usize) -> Store {
        match mode {
            StorageMode::F32 => Store::F32(alloc::vec![0.0f32; elems]),
            StorageMode::F16 => Store::F16(alloc::vec![0u16; elems]),
        }
    }

    fn elems(&self) -> usize {
        match self {
            Store::F32(v) => v.len(),
            Store::F16(v) => v.len(),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Store::F32(v) => v.len() * 4,
            Store::F16(v) => v.len() * 2,
        }
    }
}

/// Pre-allocated working memory for one max input resolution.
///
/// All buffers are allocated once in [`ScratchArena::new`]; accessors hand out
/// slices with no further allocation. Wrong-mode access panics — storage mode
/// is fixed at construction.
pub struct ScratchArena {
    mode: StorageMode,
    max_w: usize,
    max_h: usize,
    stem_c: usize,
    s8_c: usize,
    s16_c: usize,
    buf_a: Store,
    buf_b: Store,
    skip_s8: Store,
    skip_s16: Store,
    /// Per-row widening scratch for the F16 mode (one row of the widest
    /// feature map); also free for kernel-side tile staging.
    row_scratch: Vec<f32>,
}

impl ScratchArena {
    /// Allocate an arena holding activations for inputs up to
    /// `max_w`x`max_h` in `f32` mode.
    ///
    /// `stem_c` — channels of the stride-2 stem output (largest map).
    /// `s8_c`/`s16_c` — channels of the FPN skip tensors at stride 8/16.
    pub fn new(max_w: usize, max_h: usize, stem_c: usize, s8_c: usize, s16_c: usize) -> Self {
        Self::with_mode(StorageMode::F32, max_w, max_h, stem_c, s8_c, s16_c)
    }

    /// Allocate an arena storing activations as packed f16 (half footprint).
    pub fn new_f16(max_w: usize, max_h: usize, stem_c: usize, s8_c: usize, s16_c: usize) -> Self {
        Self::with_mode(StorageMode::F16, max_w, max_h, stem_c, s8_c, s16_c)
    }

    /// Full constructor with explicit storage mode.
    pub fn with_mode(
        mode: StorageMode,
        max_w: usize,
        max_h: usize,
        stem_c: usize,
        s8_c: usize,
        s16_c: usize,
    ) -> Self {
        // Ping-pong must hold the stem output (max_w/2 x max_h/2 x stem_c)
        // and every later stage's largest tensor (the stride-8 fusion).
        // NOTE: at 512x512/stem16 this is 256*256*16 = 1M elements —
        // 4 MB in f32, 2 MB per buffer in f16. The v0.2 sketch's
        // "512 KB ping-pong" figure assumed 256x256x16 x 2 B = 512 KB,
        // which is off by 4x (it is 2 MB).
        let stem_elems = (max_w / 2) * (max_h / 2) * stem_c;
        let s8_elems = (max_w / 8) * (max_h / 8) * s8_c;
        let s16_elems = (max_w / 16) * (max_h / 16) * s16_c;
        let ping_pong = stem_elems.max(s8_elems);
        ScratchArena {
            mode,
            max_w,
            max_h,
            stem_c,
            s8_c,
            s16_c,
            buf_a: Store::new(mode, ping_pong),
            buf_b: Store::new(mode, ping_pong),
            skip_s8: Store::new(mode, s8_elems),
            skip_s16: Store::new(mode, s16_elems),
            // One row of the widest feature map worth of widening scratch.
            row_scratch: alloc::vec![0.0f32; max_w / 2 * stem_c.max(s8_c)],
        }
    }

    /// Element capacity of one buffer.
    pub fn capacity(&self, id: BufId) -> usize {
        self.store(id).elems()
    }

    /// The arena's storage mode.
    pub fn storage(&self) -> StorageMode {
        self.mode
    }

    /// Configured max input dimensions and channel widths.
    pub fn dims(&self) -> (usize, usize, usize, usize, usize) {
        (self.max_w, self.max_h, self.stem_c, self.s8_c, self.s16_c)
    }

    /// `f32` working slice for a buffer (panics in `F16` mode).
    pub fn f32(&mut self, id: BufId) -> &mut [f32] {
        match self.store_mut(id) {
            Store::F32(v) => v.as_mut_slice(),
            Store::F16(_) => panic!("arena is F16; widen via f16()/expand_f16"),
        }
    }

    /// Packed-f16 storage slice for a buffer (panics in `F32` mode).
    pub fn f16(&mut self, id: BufId) -> &mut [u16] {
        match self.store_mut(id) {
            Store::F16(v) => v.as_mut_slice(),
            Store::F32(_) => panic!("arena is F32; use f32()"),
        }
    }

    /// Row/tile widening scratch (`f32`, independent of storage mode).
    pub fn row_scratch(&mut self) -> &mut [f32] {
        &mut self.row_scratch
    }

    /// Two disjoint `f32` buffer slices at once — the ping-pong pattern
    /// (`inp` in one buffer, `out` in the other) needs both simultaneously.
    /// Panics if `i == j` or the arena is `F16` storage.
    pub fn two_f32(&mut self, i: BufId, j: BufId) -> (&mut [f32], &mut [f32]) {
        assert!(i != j, "two_f32 needs two different buffers");
        fn g(s: &mut Store) -> &mut [f32] {
            match s {
                Store::F32(v) => v.as_mut_slice(),
                Store::F16(_) => panic!("arena is F16; widen via f16()/expand_f16"),
            }
        }
        let (a, b, s8, s16) = (
            &mut self.buf_a,
            &mut self.buf_b,
            &mut self.skip_s8,
            &mut self.skip_s16,
        );
        match (i, j) {
            (BufId::A, BufId::B) => (g(a), g(b)),
            (BufId::B, BufId::A) => (g(b), g(a)),
            (BufId::A, BufId::SkipS8) => (g(a), g(s8)),
            (BufId::SkipS8, BufId::A) => (g(s8), g(a)),
            (BufId::A, BufId::SkipS16) => (g(a), g(s16)),
            (BufId::SkipS16, BufId::A) => (g(s16), g(a)),
            (BufId::B, BufId::SkipS8) => (g(b), g(s8)),
            (BufId::SkipS8, BufId::B) => (g(s8), g(b)),
            (BufId::B, BufId::SkipS16) => (g(b), g(s16)),
            (BufId::SkipS16, BufId::B) => (g(s16), g(b)),
            (BufId::SkipS8, BufId::SkipS16) => (g(s8), g(s16)),
            (BufId::SkipS16, BufId::SkipS8) => (g(s16), g(s8)),
            _ => unreachable!(),
        }
    }

    /// All four `f32` buffer slices at once — graph stages whose residual
    /// input lives in a third buffer (e.g. dilated bottleneck adding the
    /// stride-16 skip) need three-way splits. Panics in `F16` mode.
    /// Order: `(A, B, SkipS8, SkipS16)`.
    pub fn parts_f32(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        fn g(s: &mut Store) -> &mut [f32] {
            match s {
                Store::F32(v) => v.as_mut_slice(),
                Store::F16(_) => panic!("arena is F16; widen via f16()/expand_f16"),
            }
        }
        (
            g(&mut self.buf_a),
            g(&mut self.buf_b),
            g(&mut self.skip_s8),
            g(&mut self.skip_s16),
        )
    }

    /// Total resident bytes of all arena buffers.
    pub fn byte_footprint(&self) -> usize {
        self.buf_a.bytes()
            + self.buf_b.bytes()
            + self.skip_s8.bytes()
            + self.skip_s16.bytes()
            + self.row_scratch.len() * 4
    }

    fn store(&self, id: BufId) -> &Store {
        match id {
            BufId::A => &self.buf_a,
            BufId::B => &self.buf_b,
            BufId::SkipS8 => &self.skip_s8,
            BufId::SkipS16 => &self.skip_s16,
        }
    }

    fn store_mut(&mut self, id: BufId) -> &mut Store {
        match id {
            BufId::A => &mut self.buf_a,
            BufId::B => &mut self.buf_b,
            BufId::SkipS8 => &mut self.skip_s8,
            BufId::SkipS16 => &mut self.skip_s16,
        }
    }
}

/// IEEE 754 binary16 -> f32 (safe, handles subnormals/inf/nan).
/// Same expansion as zensr-micro's verified version, including the corrected
/// subnormal normalization shift.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let f = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => {
            // subnormal: normalize
            let shift = m.leading_zeros() - 22;
            let m2 = (m << (shift + 1)) & 0x3ff;
            let e2 = 127 - 15 - shift;
            (sign << 31) | (e2 << 23) | (m2 << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7fc0_0000,
        (e, m) => (sign << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(f)
}

/// f32 -> IEEE 754 binary16 bits, round-to-nearest-even.
///
/// Handles subnormals and clamps overflow to +-inf; NaN collapses to the
/// canonical quiet NaN `0x7e00` (payload is not preserved).
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | if man != 0 { 0x7e00 } else { 0x7c00 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // f16 subnormal/zero: value fits only in the subnormal range.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let up = rem > halfway || (rem == halfway && (half & 1) == 1);
        return sign | (half + u32::from(up)) as u16;
    }
    let mut half = ((e as u16) << 10) | (man >> 13) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    sign | half
}

/// Widen a packed-f16 slice into an `f32` destination (row/tile staging).
pub fn expand_f16(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16_to_f32(*s);
    }
}

/// Pack an `f32` slice into binary16 storage (round-to-nearest-even).
pub fn pack_f16(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_f16(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exhaustive() {
        // Every non-NaN binary16 pattern survives f16 -> f32 -> f16.
        for bits in 0u32..=0xffff {
            let h = bits as u16;
            let exp = (h >> 10) & 0x1f;
            let man = h & 0x3ff;
            let v = f16_to_f32(h);
            if exp == 0x1f && man != 0 {
                assert!(v.is_nan(), "{h:#06x} should decode as NaN");
                continue; // NaN collapses to canonical form
            }
            let back = f32_to_f16(v);
            assert_eq!(back, h, "round-trip failed for {h:#06x} ({v})");
        }
    }

    #[test]
    fn f16_subnormal_and_specials() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x0001), f32::from_bits(0x3380_0000)); // 2^-24
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65504.0), 0x7bff); // max f16 normal
        assert_eq!(f32_to_f16(70000.0), 0x7c00); // overflow -> inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        // Smallest subnormal, and just under half of it (ties to even -> 0).
        assert_eq!(f32_to_f16(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0000)), 0x0000);
    }

    #[test]
    fn arena_footprint_and_access() {
        // 512x512, stem 16, s8 32, s16 48 (the v0.2 sketch's f16 layout).
        let mut a = ScratchArena::new_f16(512, 512, 16, 32, 48);
        assert_eq!(a.capacity(BufId::A), 256 * 256 * 16);
        assert_eq!(a.capacity(BufId::SkipS8), 64 * 64 * 32);
        assert_eq!(a.capacity(BufId::SkipS16), 32 * 32 * 48);
        // Honest f16 footprint: 2x(256*256*16*2B) + 64*64*32*2 + 32*32*48*2
        // + row_scratch(256*32*4) — the spec's "1.37 MB" underestimated the
        // ping-pong buffers by 4x.
        let expected = 2 * 256 * 256 * 16 * 2 + 64 * 64 * 32 * 2 + 32 * 32 * 48 * 2;
        let expected = expected + 256 * 32 * 4; // row_scratch
        assert_eq!(a.byte_footprint(), expected);
        assert_eq!(a.byte_footprint(), 4_587_520);
        a.f16(BufId::A)[0] = f32_to_f16(0.5);
        assert_eq!(f16_to_f32(a.f16(BufId::A)[0]), 0.5);

        let mut b = ScratchArena::new(512, 384, 16, 32, 48);
        b.f32(BufId::B)[10] = 3.25;
        assert_eq!(b.f32(BufId::B)[10], 3.25);
        assert_eq!(b.capacity(BufId::A), 256 * 192 * 16);
    }
}
