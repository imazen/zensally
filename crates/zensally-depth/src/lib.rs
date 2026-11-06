//! Optional geometric saliency cues from an existing depth map.
//!
//! Depth is not an attention label. This experimental cue measures local foreground
//! separation and must be evaluated on crop outcomes before being promoted to a default.
//! Input planes carry dimensions, row stride and an explicit near/far direction.
#![no_std]
#![forbid(unsafe_code)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;
use core::fmt;
use zensally::SaliencyMap;

#[cfg(any(feature = "jpeg", feature = "heic"))]
mod embedded;
#[cfg(any(feature = "jpeg", feature = "heic"))]
pub use embedded::{EmbeddedDepth, decode_embedded_depth};

/// Failure to interpret or process depth. Missing depth is represented separately by `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    InvalidPlane,
    InvalidOptions,
    Limit,
    Unsupported(&'static str),
    Decode(alloc::string::String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for Error {}

/// Direction of the stored samples; the cue uses relative geometry, not metric distances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DepthDirection {
    /// Larger valid samples describe farther geometry (linear distance).
    Farther,
    /// Larger valid samples describe nearer geometry (inverse distance/disparity).
    Nearer,
}

/// Borrowed scalar plane. Strides are in samples; trailing padding is never read.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    stride: usize,
}
impl<'a> Plane<'a> {
    pub fn new(data: &'a [f32], width: usize, height: usize, stride: usize) -> Result<Self, Error> {
        let required = height
            .checked_sub(1)
            .and_then(|h| h.checked_mul(stride))
            .and_then(|n| n.checked_add(width))
            .ok_or(Error::InvalidPlane)?;
        if width == 0 || stride < width || data.len() < required {
            return Err(Error::InvalidPlane);
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }
    pub fn width(self) -> usize {
        self.width
    }
    pub fn height(self) -> usize {
        self.height
    }
    pub fn row(self, y: usize) -> &'a [f32] {
        &self.data[y * self.stride..][..self.width]
    }
    fn at(self, x: usize, y: usize) -> f32 {
        self.data[y * self.stride + x]
    }
}

/// Geometry and optional per-sample confidence, aligned to the same stored RGB image.
#[derive(Clone, Copy, Debug)]
pub struct DepthView<'a> {
    pub samples: Plane<'a>,
    pub confidence: Option<Plane<'a>>,
    pub direction: DepthDirection,
    /// Maps the stored image to its display orientation; the returned map preserves it.
    pub orientation: zencodec::Orientation,
}

/// Algorithm controls, exposed for experiments rather than claimed calibrated defaults.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Working-map longest side, preserving aspect ratio and never upscaling.
    pub max_side: usize,
    /// Radius of each local foreground comparison in working-map samples.
    pub radius: usize,
    /// Positive depth difference below which geometry is treated as indistinguishable.
    pub separation: f32,
    /// Maximum number of samples admitted before any working allocations.
    pub max_input_samples: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_side: 256,
            radius: 8,
            separation: 0.02,
            max_input_samples: 64 * 1024 * 1024,
        }
    }
}

/// Saliency map plus the confidence that gates its contribution to an RGB map.
#[derive(Debug)]
pub struct DepthCue {
    pub map: SaliencyMap,
    pub confidence: Vec<f32>,
    pub orientation: zencodec::Orientation,
}

fn zeros(n: usize) -> Result<Vec<f32>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Limit)?;
    v.resize(n, 0.0);
    Ok(v)
}

/// Compute a local foreground-separation cue. Nonfinite depth and nonpositive
/// confidence contribute no evidence. A constant-depth image produces zero cue.
pub fn depth_cue(view: DepthView<'_>, options: Options) -> Result<DepthCue, Error> {
    let p = view.samples;
    let total = p.width.checked_mul(p.height).ok_or(Error::Limit)?;
    if total > options.max_input_samples {
        return Err(Error::Limit);
    }
    if options.max_side == 0
        || options.max_side > 4096
        || options.radius == 0
        || options.radius > 256
        || !options.separation.is_finite()
        || options.separation < 0.0
    {
        return Err(Error::InvalidOptions);
    }
    if view
        .confidence
        .is_some_and(|c| (c.width, c.height) != (p.width, p.height))
    {
        return Err(Error::InvalidPlane);
    }
    let scale = p.width.max(p.height);
    let w = if scale > options.max_side {
        (p.width as u64 * options.max_side as u64 / scale as u64).max(1) as usize
    } else {
        p.width
    };
    let h = if scale > options.max_side {
        (p.height as u64 * options.max_side as u64 / scale as u64).max(1) as usize
    } else {
        p.height
    };
    let n = w.checked_mul(h).ok_or(Error::Limit)?;
    let mut depth = zeros(n)?;
    let mut confidence = zeros(n)?;
    // Area resampling preserves sparse valid measurements and confidence without
    // inventing an interpolated surface across holes. Padding is excluded by row().
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f64;
            let mut weight = 0.0f64;
            let (x0, x1) = (
                x * p.width / w,
                ((x + 1) * p.width / w).max(x * p.width / w + 1),
            );
            let (y0, y1) = (
                y * p.height / h,
                ((y + 1) * p.height / h).max(y * p.height / h + 1),
            );
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let d = p.at(sx, sy);
                    let c = view.confidence.map_or(1.0, |c| c.at(sx, sy));
                    if d.is_finite() && c.is_finite() && c > 0.0 {
                        let c = c.clamp(0.0, 1.0) as f64;
                        sum += d as f64 * c;
                        weight += c;
                    }
                }
            }
            let i = y * w + x;
            if weight > 0.0 {
                depth[i] = (sum / weight) as f32;
                confidence[i] = (weight / ((x1 - x0) * (y1 - y0)) as f64) as f32;
                lo = lo.min(depth[i]);
                hi = hi.max(depth[i]);
            }
        }
    }
    let mut data = zeros(n)?;
    if hi > lo {
        let span = hi as f64 - lo as f64;
        for (d, &c) in depth.iter_mut().zip(&confidence) {
            if c > 0.0 {
                let normalized = ((*d as f64 - lo as f64) / span) as f32;
                *d = match view.direction {
                    DepthDirection::Nearer => normalized,
                    DepthDirection::Farther => 1.0 - normalized,
                };
            }
        }
        // Eight angular directions: nearer-than-surroundings evidence, with
        // confidence-weighted averaging over available directions. No center prior.
        const DIRECTIONS: [(isize, isize); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if confidence[i] == 0.0 {
                    continue;
                }
                let mut sum = 0.0;
                let mut weight = 0.0;
                for (dx, dy) in DIRECTIONS {
                    let nx = x as isize + dx * options.radius as isize;
                    let ny = y as isize + dy * options.radius as isize;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    sum += (depth[i] - depth[j] - options.separation).max(0.0) * confidence[j];
                    weight += confidence[j];
                }
                if weight > 0.0 {
                    data[i] = (sum / weight).clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(DepthCue {
        map: SaliencyMap {
            data,
            width: w as u32,
            height: h as u32,
        },
        confidence,
        orientation: view.orientation,
    })
}

/// Fuse maps in the same stored orientation, preserving RGB exactly wherever
/// depth is invalid. Map sizes may differ; depth is sampled at pixel centers.
pub fn fuse(rgb: &SaliencyMap, depth: &DepthCue, amount: f32) -> Result<SaliencyMap, Error> {
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) {
        return Err(Error::InvalidOptions);
    }
    let (w, h) = (rgb.width as usize, rgb.height as usize);
    let (dw, dh) = (depth.map.width as usize, depth.map.height as usize);
    if w == 0
        || h == 0
        || dw == 0
        || dh == 0
        || rgb.data.len() != w.checked_mul(h).ok_or(Error::Limit)?
        || depth.map.data.len() != dw.checked_mul(dh).ok_or(Error::Limit)?
        || depth.confidence.len() != depth.map.data.len()
    {
        return Err(Error::InvalidPlane);
    }
    if depth
        .confidence
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || depth.map.data.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidPlane);
    }
    let mut data = zeros(rgb.data.len())?;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let j = (((2 * y + 1) * dh / (2 * h)).min(dh - 1)) * dw
                + ((2 * x + 1) * dw / (2 * w)).min(dw - 1);
            let a = amount * depth.confidence[j].clamp(0.0, 1.0);
            data[i] = if a == 0.0 {
                rgb.data[i]
            } else {
                rgb.data[i] + a * (depth.map.data[j] - rgb.data[i])
            };
        }
    }
    Ok(SaliencyMap {
        data,
        width: rgb.width,
        height: rgb.height,
    })
}
