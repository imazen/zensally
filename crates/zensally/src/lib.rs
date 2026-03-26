#![forbid(unsafe_code)]

pub mod crop;
pub mod decode;
pub mod nms;
pub mod preprocess;

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

/// Input image for face detection.
pub struct ImageRef<'a> {
    /// Raw pixel data in the specified format.
    pub pixels: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format of the data.
    pub format: PixelFormat,
}

impl<'a> ImageRef<'a> {
    /// Create a new image reference, validating dimensions match pixel data length.
    pub fn new(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Self, &'static str> {
        let expected = width as usize * height as usize * format.bytes_per_pixel();
        if pixels.len() < expected {
            return Err("pixel buffer too small for given dimensions and format");
        }
        Ok(Self {
            pixels,
            width,
            height,
            format,
        })
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
