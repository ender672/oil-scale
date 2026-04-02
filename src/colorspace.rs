/// Pixel format understood by [`OilScale`](crate::OilScale).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorSpace {
    /// Greyscale — no sRGB gamma conversion
    G,
    /// Greyscale + alpha — premultiplied alpha
    GA,
    /// sRGB — converted to linear RGB during processing
    RGB,
    /// sRGB + alpha — sRGB-to-linear conversion and premultiplied alpha
    RGBA,
    /// sRGB without alpha — 4 bytes per pixel, 4th byte ignored
    RGBX,
    /// CMYK — no color space conversions
    CMYK,
}

impl ColorSpace {
    /// Number of components per pixel.
    pub fn components(self) -> usize {
        match self {
            ColorSpace::G => 1,
            ColorSpace::GA => 2,
            ColorSpace::RGB => 3,
            ColorSpace::RGBA | ColorSpace::RGBX | ColorSpace::CMYK => 4,
        }
    }
}
