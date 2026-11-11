//! `zensally-micro` — Phase 1 inference core for the zensally v0.2 rewrite.
//!
//! Static NHWC (interleaved channels) micro-graph building blocks over
//! `archmage` + `magetypes`, a pre-allocated ping-pong `ScratchArena`
//! (zero heap allocations per warm forward pass), and a CenterNet-style
//! 3x3 local-maximum peak decoder for the stride-8 multi-task head.
//!
//! `#![forbid(unsafe_code)]`, `no_std + alloc` (`std` is a default feature).
//!
//! Channel counts that flow through vector kernels are multiples of 8
//! (`8, 16, 24, 32, 48, 64, 72`): one `f32x8` is exactly one channel-vector,
//! so depthwise taps are 9 contiguous FMAs with zero shuffles or gathers.

#![no_std]
#![forbid(unsafe_code)]
// Same lint posture as zensr-micro: kernel signatures take shape as scalars
// on purpose (constants visible at the call site); indexed loops over fixed
// arrays are the pattern LLVM vectorises here.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod arena;
pub mod kernels;

pub use arena::{BufId, ScratchArena, StorageMode};
pub use kernels::{
    Activation, FpnDemoWeights, Peak, bilinear_upsample2x_add_nhwc, centernet_peaks_3x3,
    conv3x3_s2_nhwc, depthwise_conv3x3_nhwc, fpn_demo_forward, pack_depthwise_w, pack_pointwise_w,
    pack_stem_w, pointwise_conv1x1_nhwc, pointwise_conv1x1_residual_nhwc, se_gate_nhwc,
};
