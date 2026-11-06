use crate::{DepthDirection, DepthView, Error, Plane, zeros};
use alloc::{format, vec::Vec};
use zencodec::{ImageFormat, Orientation, ResourceLimits};

/// Decoded embedded geometry in its native image orientation. Samples are either
/// distances in the declared JPEG units or relative HEIC/MPF geometry; `direction`
/// explicitly records which way is nearer. This is not a color image.
#[derive(Debug)]
pub struct EmbeddedDepth {
    pub samples: Vec<f32>,
    pub confidence: Option<Vec<f32>>,
    pub width: usize,
    pub height: usize,
    pub primary_width: u32,
    pub primary_height: u32,
    pub orientation: Orientation,
    pub direction: DepthDirection,
    pub source: &'static str,
}
impl EmbeddedDepth {
    pub fn view(&self) -> Result<DepthView<'_>, Error> {
        Ok(DepthView {
            samples: Plane::new(&self.samples, self.width, self.height, self.width)?,
            confidence: self
                .confidence
                .as_ref()
                .map(|c| Plane::new(c, self.width, self.height, self.width))
                .transpose()?,
            direction: self.direction,
            orientation: self.orientation,
        })
    }
}
fn decode_error(e: impl core::fmt::Display) -> Error {
    Error::Decode(format!("{e}"))
}

/// Extract existing depth using the codecs' native extraction APIs. A supported
/// file without depth returns `None`; malformed or unsupported depth returns an
/// error. The caller selects the format through its normal zencodec registry.
pub fn decode_embedded_depth(
    data: &[u8],
    format: ImageFormat,
    limits: &ResourceLimits,
) -> Result<Option<EmbeddedDepth>, Error> {
    limits
        .check_input_size(data.len() as u64)
        .map_err(|_| Error::Limit)?;
    match format {
        #[cfg(feature = "jpeg")]
        ImageFormat::Jpeg => jpeg(data, limits),
        #[cfg(feature = "heic")]
        ImageFormat::Heic => heic(data, limits),
        _ => Err(Error::Unsupported("embedded depth carrier")),
    }
}

#[cfg(feature = "jpeg")]
fn grayscale(
    data: &[u8],
    mime: &str,
    limits: &ResourceLimits,
) -> Result<(Vec<f32>, usize, usize), Error> {
    if mime == "image/jpeg" {
        use zenjpeg::decode::{Decoder, OutputTarget, PixelFormat};
        let dec = Decoder::new()
            .output_format(PixelFormat::Gray)
            .output_target(OutputTarget::Native16)
            .max_pixels(limits.max_pixels.unwrap_or(64 * 1024 * 1024))
            .max_memory(limits.max_memory_bytes.unwrap_or(512 * 1024 * 1024));
        let info = dec.read_info(data).map_err(decode_error)?;
        limits
            .check_dimensions(info.dimensions.width, info.dimensions.height)
            .map_err(|_| Error::Limit)?;
        if info.num_components != 1 {
            return Err(Error::Unsupported("depth JPEG must be grayscale"));
        }
        let image = dec
            .decode(data, enough::Unstoppable)
            .map_err(decode_error)?;
        let max = ((1u32 << image.sample_precision()) - 1) as f32;
        let (w, h) = (image.width as usize, image.height as usize);
        let raw = image
            .pixels_u16()
            .ok_or(Error::Unsupported("native integer JPEG depth"))?;
        let mut samples = zeros(w.checked_mul(h).ok_or(Error::Limit)?)?;
        if raw.len() != samples.len() {
            return Err(Error::InvalidPlane);
        }
        for (d, &v) in samples.iter_mut().zip(raw) {
            *d = v as f32 / max;
        }
        Ok((samples, w, h))
    } else if mime == "image/png" {
        let info = zenpng::probe(data).map_err(decode_error)?;
        limits
            .check_dimensions(info.width, info.height)
            .map_err(|_| Error::Limit)?;
        let mut cfg = zenpng::PngDecodeConfig::strict();
        cfg.max_pixels = limits.max_pixels;
        cfg.max_memory_bytes = limits.max_memory_bytes;
        let image = zenpng::decode(data, &cfg, &enough::Unstoppable).map_err(decode_error)?;
        let p = image.pixels.as_slice();
        if !image.pixels.is_grayscale() {
            return Err(Error::Unsupported("depth PNG must be grayscale"));
        }
        let (w, h) = (p.width() as usize, p.rows() as usize);
        let channels = p.descriptor().channels();
        let mut samples = zeros(w.checked_mul(h).ok_or(Error::Limit)?)?;
        for y in 0..h {
            let row = p.row(y as u32);
            for x in 0..w {
                samples[y * w + x] = match p.descriptor().channel_type() {
                    zenpixels::ChannelType::U8 => row[x * channels] as f32 / 255.0,
                    zenpixels::ChannelType::U16 => {
                        let i = x * channels * 2;
                        u16::from_ne_bytes([row[i], row[i + 1]]) as f32 / 65535.0
                    }
                    _ => return Err(Error::Unsupported("depth PNG sample type")),
                };
            }
        }
        Ok((samples, w, h))
    } else {
        Err(Error::Unsupported("depth image MIME type"))
    }
}

#[cfg(feature = "jpeg")]
fn jpeg(data: &[u8], limits: &ResourceLimits) -> Result<Option<EmbeddedDepth>, Error> {
    use zenjpeg::decode::{Decoder, GDepthFormat, PixelFormat, PreserveConfig};
    let dec = Decoder::new()
        .preserve(PreserveConfig::all())
        .output_format(PixelFormat::Gray)
        .max_pixels(limits.max_pixels.unwrap_or(64 * 1024 * 1024))
        .max_memory(limits.max_memory_bytes.unwrap_or(512 * 1024 * 1024));
    let info = dec.read_info(data).map_err(decode_error)?;
    limits
        .check_dimensions(info.dimensions.width, info.dimensions.height)
        .map_err(|_| Error::Limit)?;
    let orientation = info
        .exif
        .as_deref()
        .and_then(zencodec::Exif::parse)
        .and_then(|e| e.orientation())
        .unwrap_or_default();
    let image = dec
        .decode(data, enough::Unstoppable)
        .map_err(decode_error)?;
    let Some(extras) = image.extras() else {
        return Ok(None);
    };
    let Some(depth) = extras.extract_depth_map(Some(data)) else {
        return if extras.has_depth_map() {
            Err(Error::Unsupported("declared depth could not be extracted"))
        } else {
            Ok(None)
        };
    };
    let (mut samples, w, h) = grayscale(&depth.data, &depth.mime, limits)?;
    let direction = if let Some(m) = &depth.metadata {
        if !m.near.is_finite() || !m.far.is_finite() || m.near < 0.0 || m.far <= m.near {
            return Err(Error::Unsupported("invalid or missing depth range"));
        }
        for q in &mut samples {
            // Google RangeInverse is zero at near, one at far. It is not the
            // reversed inverse-distance formula in the legacy codec comment.
            *q = match m.format {
                GDepthFormat::RangeLinear => m.near + *q * (m.far - m.near),
                GDepthFormat::RangeInverse => {
                    if m.near == 0.0 {
                        return Err(Error::Unsupported("inverse depth needs positive near"));
                    }
                    (m.far as f64 * m.near as f64
                        / (m.far as f64 - *q as f64 * (m.far - m.near) as f64))
                        as f32
                }
            };
        }
        DepthDirection::Farther
    } else {
        DepthDirection::Nearer
    };
    let confidence = depth
        .confidence
        .as_ref()
        .map(|bytes| {
            let (samples, cw, ch) = grayscale(
                bytes,
                depth.confidence_mime.as_deref().unwrap_or("image/png"),
                limits,
            )?;
            if (cw, ch) != (w, h) {
                return Err(Error::InvalidPlane);
            }
            Ok(samples)
        })
        .transpose()?;
    Ok(Some(EmbeddedDepth {
        samples,
        confidence,
        width: w,
        height: h,
        primary_width: info.dimensions.width,
        primary_height: info.dimensions.height,
        orientation,
        direction,
        source: "jpeg-embedded",
    }))
}

#[cfg(feature = "heic")]
fn heic(data: &[u8], limits: &ResourceLimits) -> Result<Option<EmbeddedDepth>, Error> {
    let info = heic::ImageInfo::from_bytes(data).map_err(decode_error)?;
    limits
        .check_dimensions(info.width, info.height)
        .map_err(|_| Error::Limit)?;
    if !info.has_depth {
        return Ok(None);
    }
    let depth = heic::DecoderConfig::new()
        .decode_depth(data)
        .map_err(decode_error)?;
    limits
        .check_dimensions(depth.width, depth.height)
        .map_err(|_| Error::Limit)?;
    let direction = match depth.depth_info.representation_type {
        heic::DepthRepresentationType::UniformZ => DepthDirection::Farther,
        heic::DepthRepresentationType::UniformInverseZ
        | heic::DepthRepresentationType::UniformDisparity => DepthDirection::Nearer,
        _ => return Err(Error::Unsupported("nonuniform HEIC depth representation")),
    };
    let (w, h) = (depth.width as usize, depth.height as usize);
    if depth.bit_depth == 0 || depth.bit_depth > 16 {
        return Err(Error::InvalidPlane);
    }
    let max = ((1u32 << depth.bit_depth) - 1) as f32;
    let mut samples = zeros(w.checked_mul(h).ok_or(Error::Limit)?)?;
    if samples.len() != depth.data.len() {
        return Err(Error::InvalidPlane);
    }
    for (d, &v) in samples.iter_mut().zip(&depth.data) {
        *d = v as f32 / max;
    }
    Ok(Some(EmbeddedDepth {
        samples,
        confidence: None,
        width: w,
        height: h,
        primary_width: info.width,
        primary_height: info.height,
        orientation: Orientation::Identity,
        direction,
        source: "heic-auxiliary",
    }))
}
