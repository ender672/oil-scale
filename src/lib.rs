pub mod colorspace;
pub mod srgb;
pub mod kernel;
pub mod scale;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "png")]
pub mod png;
#[cfg(feature = "jpeg-turbo")]
pub mod jpeg_ffi;
#[cfg(target_arch = "x86_64")]
pub mod sse2;

pub use colorspace::ColorSpace;
pub use scale::{OilScale, OilError, fix_ratio};
