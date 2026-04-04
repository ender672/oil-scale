#![warn(missing_docs)]

//! Fast, streaming image resizer with sRGB-correct interpolation and low
//! memory usage.
//!
//! `oil-scale` scales images one scanline at a time using separable convolution,
//! keeping only the rows needed for the current output line in memory. All
//! interpolation is performed in linear light with premultiplied alpha to
//! produce correct results.
//!
//! # Quick start
//!
//! ```
//! use oil_scale::{OilScale, ColorSpace};
//!
//! // Set up a scaler: 1920×1080 input → 640×360 output, RGB
//! let mut scaler = OilScale::new(1920, 1080, 640, 360, ColorSpace::RGB).unwrap();
//! let mut out_row = vec![0u8; 640 * 3];
//!
//! for _ in 0..360 {
//!     for _ in 0..scaler.slots() {
//!         let input_scanline: Vec<u8> = vec![0; 1920 * 3]; // read from your source
//!         scaler.push_scanline(&input_scanline).unwrap();
//!     }
//!     scaler.read_scanline(&mut out_row);
//! }
//! ```
//!
//! Enable the `png` and `jpeg` cargo features (on by default) for built-in
//! codec helpers, or use the core scaler alone with `--no-default-features`.

/// Color space definitions used by the scaler.
pub mod colorspace;
/// Core streaming scaler.
pub mod scale;
/// JPEG codec helpers.
#[cfg(feature = "jpeg")]
pub mod jpeg;
/// PNG codec helpers.
#[cfg(feature = "png")]
pub mod png;
/// libjpeg-turbo FFI bindings.
#[cfg(feature = "jpeg-turbo")]
pub mod jpeg_ffi;

pub(crate) mod srgb;
pub(crate) mod kernel;
#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
pub(crate) mod sse2;

pub use colorspace::ColorSpace;
pub use scale::{OilScale, Error, fix_ratio};
