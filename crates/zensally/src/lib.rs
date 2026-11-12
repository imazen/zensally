#![forbid(unsafe_code)]

pub mod crop;
pub mod decode;
pub mod metadata;
pub mod nms;
pub mod preprocess;

pub mod sat_crop;
pub mod stream_prep;

pub use metadata::{CropMetadata, CuePlane, MetadataError, SubjectKind, SubjectRegion};
pub use stream_prep::JpegDctBlocks;

#[cfg(feature = "zenlayout")]
pub mod bridge;

/// A detected face region with confidence score.
///
/// Coordinates are percentages (0.0–100.0) of image dimensions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceRect {
    /// Left edge as percentage of image width.
    pub x1: f32,
    /// Top edge as percentage of image height.
    pub y1: f32,
    /// Right edge as percentage of image width.
    pub x2: f32,
    /// Bottom edge as percentage of image height.
    pub y2: f32,
    /// Detection confidence (0.0–1.0).
    pub confidence: f32,
}

/// Pixel format of input image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 4 bytes per pixel: blue, green, red, alpha.
    Bgra,
    /// 3 bytes per pixel: red, green, blue.
    Rgb,
    /// 4 bytes per pixel: red, green, blue, alpha.
    Rgba,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Bgra | Self::Rgba => 4,
            Self::Rgb => 3,
        }
    }
}

/// Input image for face and saliency detection.
#[derive(Debug, Clone, Copy)]
pub struct ImageRef<'a> {
    /// Raw pixel data in the specified format.
    pub pixels: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row stride in bytes (`>= width as usize * format.bytes_per_pixel()`).
    pub stride: usize,
    /// Pixel format of the data.
    pub format: PixelFormat,
}

impl<'a> ImageRef<'a> {
    /// Create a tightly packed image reference (`stride = width * format.bytes_per_pixel()`),
    /// validating that `pixels` is large enough for the given dimensions and format.
    pub fn new(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Self, &'static str> {
        let stride = width as usize * format.bytes_per_pixel();
        Self::new_strided(pixels, width, height, stride, format)
    }

    /// Create a strided image reference, validating that `stride >= width * bpp`
    /// and that `pixels` covers all `height` rows.
    pub fn new_strided(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
    ) -> Result<Self, &'static str> {
        let row_bytes = width as usize * format.bytes_per_pixel();
        if stride < row_bytes {
            return Err("row stride smaller than width * bytes_per_pixel");
        }
        let expected = if height == 0 {
            0
        } else {
            (height as usize - 1) * stride + row_bytes
        };
        if pixels.len() < expected {
            return Err("pixel buffer too small for given dimensions and format");
        }
        Ok(Self {
            pixels,
            width,
            height,
            stride,
            format,
        })
    }

    /// Convert any 8-bit RGB, RGBA/RGBX, or BGRA/BGRX [`zenpixels::PixelSlice`] into
    /// a zero-copy strided [`ImageRef`].
    #[cfg(feature = "zenpixels")]
    pub fn from_pixel_slice<P>(slice: zenpixels::PixelSlice<'a, P>) -> Result<Self, &'static str> {
        let desc = slice.descriptor();
        if desc.channel_type() != zenpixels::ChannelType::U8 {
            return Err("ImageRef only supports 8-bit (U8) pixel slices");
        }
        let format = match desc.layout() {
            zenpixels::ChannelLayout::Rgb => PixelFormat::Rgb,
            zenpixels::ChannelLayout::Rgba => PixelFormat::Rgba,
            zenpixels::ChannelLayout::Bgra => PixelFormat::Bgra,
            _ => return Err("ImageRef only supports Rgb, Rgba/Rgbx, or Bgra/Bgrx layouts"),
        };
        Ok(Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format,
        })
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, rgb::Rgb<u8>>> for ImageRef<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, rgb::Rgb<u8>>) -> Self {
        Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format: PixelFormat::Rgb,
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, rgb::Rgba<u8>>> for ImageRef<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, rgb::Rgba<u8>>) -> Self {
        Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format: PixelFormat::Rgba,
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, rgb::alt::BGRA<u8>>> for ImageRef<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, rgb::alt::BGRA<u8>>) -> Self {
        Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format: PixelFormat::Bgra,
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, zenpixels::Rgbx>> for ImageRef<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, zenpixels::Rgbx>) -> Self {
        Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format: PixelFormat::Rgba,
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, zenpixels::Bgrx>> for ImageRef<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, zenpixels::Bgrx>) -> Self {
        Self {
            pixels: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
            format: PixelFormat::Bgra,
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> TryFrom<zenpixels::PixelSlice<'a>> for ImageRef<'a> {
    type Error = &'static str;

    fn try_from(slice: zenpixels::PixelSlice<'a>) -> Result<Self, Self::Error> {
        Self::from_pixel_slice(slice)
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> TryFrom<ImageRef<'a>> for zenpixels::PixelSlice<'a> {
    type Error = zenpixels::BufferError;

    fn try_from(image: ImageRef<'a>) -> Result<Self, Self::Error> {
        let desc = match image.format {
            PixelFormat::Rgb => zenpixels::PixelDescriptor::RGB8_SRGB,
            PixelFormat::Rgba => zenpixels::PixelDescriptor::RGBA8_SRGB,
            PixelFormat::Bgra => zenpixels::PixelDescriptor::BGRA8_SRGB,
        };
        zenpixels::PixelSlice::new(image.pixels, image.width, image.height, image.stride, desc)
            .map_err(|e| e.decompose().0)
    }
}

/// Face detector backend.
pub trait FaceDetector {
    /// Detect faces in the given image.
    ///
    /// Returns a list of face rectangles sorted by confidence (highest first).
    fn detect(&mut self, image: &ImageRef<'_>) -> Vec<FaceRect>;
}

/// Saliency detector backend.
pub trait SaliencyDetector {
    /// Compute a saliency heatmap for the given image.
    ///
    /// Returns a flat array of `width * height` values in \[0.0, 1.0\],
    /// row-major, where 1.0 = most salient.
    /// The output dimensions match the model's native resolution, not the input image.
    fn saliency_map(&mut self, image: &ImageRef<'_>) -> SaliencyMap;
}

/// A saliency heatmap at the model's native resolution.
pub struct SaliencyMap {
    /// Saliency values in \[0.0, 1.0\], row-major.
    pub data: Vec<f32>,
    /// Width of the heatmap.
    pub width: u32,
    /// Height of the heatmap.
    pub height: u32,
}

/// Full detection output including raw saliency data.
///
/// Used internally to pass results from detectors to smart crop computation.
/// For serializable metadata, convert to [`DetectionSummary`].
pub struct AnalysisOutput {
    /// Detected faces (percentage coordinates, sorted by confidence).
    pub faces: Vec<FaceRect>,
    /// Full saliency heatmap at model resolution.
    pub saliency: Option<SaliencyMap>,
}

impl AnalysisOutput {
    /// Create a serializable summary (without raw saliency data).
    pub fn summary(&self) -> DetectionSummary {
        DetectionSummary {
            faces: self.faces.clone(),
            saliency_dims: self.saliency.as_ref().map(|s| (s.width, s.height)),
        }
    }
}

/// Serializable summary of detection results (no raw pixel data).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectionSummary {
    /// Detected faces (percentage coordinates, sorted by confidence).
    pub faces: Vec<FaceRect>,
    /// Saliency heatmap dimensions (width, height), if computed.
    pub saliency_dims: Option<(u32, u32)>,
}

/// Crop rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Serializable record of a smart crop decision.
///
/// Captures what was detected, what crop was chosen, and the parameters
/// used — useful for debugging, UI overlays, and logging.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SmartCropResult {
    /// Detection results (faces found, saliency dims).
    pub detection: DetectionSummary,
    /// The chosen crop rectangle in source pixels, or None if no crop was applied.
    pub crop: Option<CropRect>,
    /// Requested target aspect ratio (w, h).
    pub target_aspect: (u32, u32),
    /// Crop mode used ("minimal" or "maximal").
    pub mode: String,
    /// User-supplied manual focus regions, if any.
    pub manual_focus: Vec<FocusRegion>,
}

/// A user-specified focus region (from `&focus=` parameter).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FocusRegion {
    /// Left edge as percentage of image width (0.0–100.0).
    pub x1: f32,
    /// Top edge as percentage of image height (0.0–100.0).
    pub y1: f32,
    /// Right edge as percentage of image width (0.0–100.0).
    pub x2: f32,
    /// Bottom edge as percentage of image height (0.0–100.0).
    pub y2: f32,
}

/// Serializable record of a whitespace crop decision.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WhitespaceCropResult {
    /// Original source dimensions before trimming.
    pub original: (u32, u32),
    /// Detected content bounds.
    pub content_bounds: CropRect,
    /// Threshold used for detection.
    pub threshold: u8,
    /// Padding percentage applied.
    pub padding_applied: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_ref_new_and_strided_validation() {
        let packed = [0u8; 4 * 2 * 3];
        let img = ImageRef::new(&packed, 4, 2, PixelFormat::Rgb).unwrap();
        assert_eq!(img.stride, 12);

        // Strided buffer where last row has no trailing stride padding
        let strided = [0u8; 16 + 12];
        let img_s = ImageRef::new_strided(&strided, 4, 2, 16, PixelFormat::Rgb).unwrap();
        assert_eq!(img_s.stride, 16);

        assert!(ImageRef::new_strided(&strided, 4, 2, 10, PixelFormat::Rgb).is_err());
        assert!(ImageRef::new_strided(&strided[..27], 4, 2, 16, PixelFormat::Rgb).is_err());
    }

    #[cfg(feature = "zenpixels")]
    #[test]
    fn zenpixels_bridge_roundtrip_strided() {
        let mut raw = vec![0u8; 24 * 4]; // 4 rows, stride 24 (multiple of 3), width 6 RGB (18 bytes/row)
        for y in 0..4usize {
            for x in 0..6usize {
                let off = y * 24 + x * 3;
                raw[off] = (x * 20) as u8;
                raw[off + 1] = (y * 40) as u8;
                raw[off + 2] = 128;
            }
        }
        let slice =
            zenpixels::PixelSlice::new(&raw, 6, 4, 24, zenpixels::PixelDescriptor::RGB8_SRGB)
                .unwrap();
        let typed = slice.try_typed::<rgb::Rgb<u8>>().unwrap();
        let img = ImageRef::from(typed);
        assert_eq!(img.width, 6);
        assert_eq!(img.height, 4);
        assert_eq!(img.stride, 24);
        assert_eq!(img.format, PixelFormat::Rgb);

        let erased = zenpixels::PixelSlice::try_from(img).unwrap();
        assert_eq!(erased.width(), 6);
        assert_eq!(erased.rows(), 4);
        assert_eq!(erased.stride(), 24);

        let img2 = ImageRef::try_from(erased).unwrap();
        assert_eq!(img2.stride, 24);
        assert_eq!(img2.format, PixelFormat::Rgb);
    }
}
