use std::sync::OnceLock;

const L2S_LEN: usize = 32768;

/// Precomputed lookup tables for sRGB ↔ linear conversions.
///
/// The linear-to-sRGB LUT covers the full [0, 1] linear range; callers must
/// clamp their input before indexing. Catmull-Rom's negative lobe can drive
/// intermediates outside [0, 1] in 2-D, and RGBA/ARGB unpremul (R_pre / alpha
/// as alpha approaches 0) can produce arbitrarily large indices, so every
/// gamma-aware output path clamps before the lookup.
pub struct SrgbTables {
    /// sRGB byte -> linear float (gamma decompression)
    pub s2l: [f32; 256],
    /// byte -> float identity mapping (no gamma, for greyscale/CMYK)
    pub i2f: [f32; 256],
    /// linear-to-sRGB table; full [0, 1] mapping, no padding.
    l2s: [u8; L2S_LEN],
    /// Length of the l2s mapping (always L2S_LEN).
    pub l2s_len: usize,
}

impl SrgbTables {
    fn build() -> Self {
        let mut tables = SrgbTables {
            s2l: [0.0; 256],
            i2f: [0.0; 256],
            l2s: [0; L2S_LEN],
            l2s_len: L2S_LEN,
        };

        // build s2l: sRGB byte -> linear float
        for input in 0..=255u16 {
            let in_f = input as f64 / 255.0;
            let val = if in_f <= 0.040448236277 {
                in_f / 12.92
            } else {
                ((in_f + 0.055) / 1.055).powf(2.4)
            };
            tables.s2l[input as usize] = val as f32;
        }

        // build i2f: identity byte -> float
        for i in 0..=255u16 {
            tables.i2f[i as usize] = i as f32 / 255.0;
        }

        // build l2s: linear float -> sRGB byte (full [0, 1] range)
        for i in 0..L2S_LEN {
            let srgb_f = (i as f64 + 0.5) / (L2S_LEN - 1) as f64;
            let val = if srgb_f <= 0.00313 {
                srgb_f * 12.92
            } else {
                1.055 * srgb_f.powf(1.0 / 2.4) - 0.055
            };
            tables.l2s[i] = (val * 255.0).round() as u8;
        }

        tables
    }

    /// Map a linear RGB float to an sRGB byte. Clamps the input to [0, 1]
    /// internally so callers don't have to.
    #[inline]
    pub fn linear_to_srgb(&self, val: f32) -> u8 {
        let v = if val < 0.0 {
            0.0
        } else if val > 1.0 {
            1.0
        } else {
            val
        };
        let idx = (v * (L2S_LEN - 1) as f32) as usize;
        self.l2s[idx]
    }

    /// Pointer to the start of the l2s LUT. SIMD callers index this directly,
    /// and must clamp their float vector to [0, 1] before the cvtps-to-int.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline]
    pub fn l2s_ptr(&self) -> *const u8 {
        self.l2s.as_ptr()
    }
}

static TABLES: OnceLock<SrgbTables> = OnceLock::new();

/// Return the global sRGB lookup tables, initializing them on first call.
pub fn tables() -> &'static SrgbTables {
    TABLES.get_or_init(SrgbTables::build)
}
