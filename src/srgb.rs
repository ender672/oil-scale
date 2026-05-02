use std::sync::OnceLock;

// LUT max quantization error scales as ~1647/(L-1): at L=32768 it was 0.050
// (255-unit), at L=22000 it's 0.075. End-to-end tests still pass against the
// 0.07 tolerance because the pipeline's non-LUT error floor dominates once L
// exceeds ~22000. Frees ~10 KB of L1d for the float accumulator buffer and row
// data to coexist without evicting LUT sets.
const L2S_LEN: usize = 22000;

// Each LUT is its own static, mirroring C's separate `s2l_map` / `i2f_map` /
// `l2s_map_storage` globals. Bundling them into a single struct ties the
// cache placement of the small (s2l/i2f) tables to the size of the large
// (l2s) one — so any future change to L2S_LEN perturbs hot reads on
// gamma-aware paths.
static S2L_DATA: OnceLock<[f32; 256]> = OnceLock::new();
static I2F_DATA: OnceLock<[f32; 256]> = OnceLock::new();
static L2S_DATA: OnceLock<[u8; L2S_LEN]> = OnceLock::new();

/// Precomputed lookup tables for sRGB ↔ linear conversions.
///
/// The linear-to-sRGB LUT covers the full [0, 1] linear range; callers must
/// clamp their input before indexing. Catmull-Rom's negative lobe can drive
/// intermediates outside [0, 1] in 2-D, and RGBA/ARGB unpremul (R_pre / alpha
/// as alpha approaches 0) can produce arbitrarily large indices, so every
/// gamma-aware output path clamps before the lookup.
pub struct SrgbTables {
    /// sRGB byte -> linear float (gamma decompression)
    pub s2l: &'static [f32; 256],
    /// byte -> float identity mapping (no gamma, for greyscale/CMYK)
    pub i2f: &'static [f32; 256],
    /// linear-to-sRGB table; full [0, 1] mapping, no padding.
    l2s: &'static [u8; L2S_LEN],
    /// Length of the l2s mapping (always L2S_LEN).
    pub l2s_len: usize,
}

impl SrgbTables {
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

fn build_s2l() -> [f32; 256] {
    let mut arr = [0.0f32; 256];
    for input in 0..=255u16 {
        let in_f = input as f64 / 255.0;
        let val = if in_f <= 0.040448236277 {
            in_f / 12.92
        } else {
            ((in_f + 0.055) / 1.055).powf(2.4)
        };
        arr[input as usize] = val as f32;
    }
    arr
}

fn build_i2f() -> [f32; 256] {
    let mut arr = [0.0f32; 256];
    for i in 0..=255u16 {
        arr[i as usize] = i as f32 / 255.0;
    }
    arr
}

fn build_l2s() -> [u8; L2S_LEN] {
    let mut arr = [0u8; L2S_LEN];
    for i in 0..L2S_LEN {
        let srgb_f = (i as f64 + 0.5) / (L2S_LEN - 1) as f64;
        let val = if srgb_f <= 0.00313 {
            srgb_f * 12.92
        } else {
            1.055 * srgb_f.powf(1.0 / 2.4) - 0.055
        };
        arr[i] = (val * 255.0).round() as u8;
    }
    arr
}

static TABLES_HANDLE: OnceLock<SrgbTables> = OnceLock::new();

/// Return the global sRGB lookup tables, initializing them on first call.
pub fn tables() -> &'static SrgbTables {
    TABLES_HANDLE.get_or_init(|| SrgbTables {
        s2l: S2L_DATA.get_or_init(build_s2l),
        i2f: I2F_DATA.get_or_init(build_i2f),
        l2s: L2S_DATA.get_or_init(build_l2s),
        l2s_len: L2S_LEN,
    })
}
