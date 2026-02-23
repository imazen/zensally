#![forbid(unsafe_code)]

/// A detected face region with confidence score.
///
/// Coordinates are percentages (0.0–100.0) of image dimensions.
#[derive(Debug, Clone, PartialEq)]
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
