use crate::colorspace::ColorSpace;
use crate::kernel;
use crate::srgb;
#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
use crate::sse2;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use crate::neon;

const MAX_DIMENSION: u32 = 1_000_000;
const TAPS: usize = 4;

/// Errors returned by `oil-scale` operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A parameter was out of range or otherwise invalid (e.g. zero dimensions,
    /// dimensions exceeding the 1,000,000 limit, or mismatched scale directions).
    InvalidArgument,
    /// An internal buffer allocation failed.
    AllocationFailed,
    /// An I/O operation failed.
    Io(std::io::Error),
    /// An image codec operation failed (encoding or decoding).
    Codec(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidArgument => write!(f, "invalid argument"),
            Error::AllocationFailed => write!(f, "allocation failed"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Codec(e) => write!(f, "codec error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Codec(e) => Some(&**e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Streaming image scaler that processes one scanline at a time.
#[derive(Debug)]
#[must_use]
pub struct OilScale {
    in_height: u32,
    out_height: u32,
    in_width: u32,
    out_width: u32,
    cs: ColorSpace,
    in_pos: u32,
    out_pos: u32,
    coeffs_y: Vec<f32>,
    coeffs_x: Vec<f32>,
    borders_x: Vec<i32>,
    borders_y: Vec<i32>,
    sums_y: Vec<f32>,
    sums_y_tap: usize,
    rb: Vec<f32>,
    tmp_coeffs: Vec<f32>,
    is_upscale: bool,
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
#[inline]
fn clampf(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
#[inline]
fn f2i(x: f32) -> u8 {
    (x + 0.5) as u8
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
#[inline]
fn add_sample_to_sum(sample: f32, coeffs: &[f32], sum: &mut [f32]) {
    for i in 0..4 {
        sum[i] += sample * coeffs[i];
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
#[inline]
fn push_f(f: &mut [f32; 4], val: f32) {
    f[0] = f[1];
    f[1] = f[2];
    f[2] = f[3];
    f[3] = val;
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
#[inline]
fn shift_left(f: &mut [f32]) {
    f[0] = f[1];
    f[1] = f[2];
    f[2] = f[3];
    f[3] = 0.0;
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_g(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let mut smp = [0.0f32; 4];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        push_f(&mut smp, input[i] as f32 / 255.0);
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            out[out_idx] = smp[0] * coeffs[0]
                + smp[1] * coeffs[1]
                + smp[2] * coeffs[2]
                + smp[3] * coeffs[3];
            out_idx += 1;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_g(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    for i in 0..len {
        let sum = coeffs[0] * lines[0][i]
            + coeffs[1] * lines[1][i]
            + coeffs[2] * lines[2][i]
            + coeffs[3] * lines[3][i];
        out[i] = f2i(clampf(sum) * 255.0);
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_g(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let mut sum = [0.0f32; 4];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            add_sample_to_sum(input[in_idx] as f32 / 255.0, cx, &mut sum);
            in_idx += 1;
            cx_idx += 4;
        }

        add_sample_to_sum(sum[0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
        shift_left(&mut sum);
        sy_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_g(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let mut s_idx = 0usize;
    for i in 0..sl_len {
        out[i] = f2i(clampf(sums[s_idx]) * 255.0);
        shift_left(&mut sums[s_idx..s_idx + 4]);
        s_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_ga(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let mut smp = [[0.0f32; 4]; 2]; // gray, alpha
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 2;
        let alpha = input[in_base + 1] as f32 / 255.0;
        push_f(&mut smp[1], alpha);
        let premul = smp[1][3] * (input[in_base] as f32 / 255.0);
        push_f(&mut smp[0], premul);
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..2 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 2;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_ga(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let mut i = 0;
    while i < len {
        let mut sums = [0.0f32; 2];
        for j in 0..2 {
            sums[j] = coeffs[0] * lines[0][i + j]
                + coeffs[1] * lines[1][i + j]
                + coeffs[2] * lines[2][i + j]
                + coeffs[3] * lines[3][i + j];
        }
        let alpha = clampf(sums[1]);
        if alpha != 0.0 {
            sums[0] /= alpha;
        }
        out[i] = f2i(clampf(sums[0]) * 255.0);
        out[i + 1] = f2i(alpha * 255.0);
        i += 2;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_ga(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let mut sum = [[0.0f32; 4]; 2]; // gray, alpha
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            let alpha = input[in_idx + 1] as f32 / 255.0;
            add_sample_to_sum(input[in_idx] as f32 / 255.0 * alpha, cx, &mut sum[0]);
            add_sample_to_sum(alpha, cx, &mut sum[1]);
            in_idx += 2;
            cx_idx += 4;
        }

        for j in 0..2 {
            add_sample_to_sum(sum[j][0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_ga(sums: &mut [f32], width: usize, out: &mut [u8]) {
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let alpha = clampf(sums[s_idx + 4]);
        if alpha != 0.0 {
            sums[s_idx] /= alpha;
        }
        out[o_idx] = f2i(clampf(sums[s_idx]) * 255.0);
        shift_left(&mut sums[s_idx..s_idx + 4]);
        out[o_idx + 1] = f2i(alpha * 255.0);
        shift_left(&mut sums[s_idx + 4..s_idx + 8]);
        s_idx += 8;
        o_idx += 2;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgb(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 3];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;
        for j in 0..3 {
            push_f(&mut smp[j], tables.s2l[input[in_base + j] as usize]);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..3 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_rgb(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    for i in 0..len {
        let sum = coeffs[0] * lines[0][i]
            + coeffs[1] * lines[1][i]
            + coeffs[2] * lines[2][i]
            + coeffs[3] * lines[3][i];
        out[i] = tables.linear_to_srgb(sum);
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgba(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 4]; // R, G, B, A
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha = input[in_base + 3] as f32 / 255.0;
        push_f(&mut smp[3], alpha);
        let alpha_val = smp[3][3];
        for j in 0..3 {
            push_f(&mut smp[j], alpha_val * tables.s2l[input[in_base + j] as usize]);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..4 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_rgba(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let mut i = 0;
    while i < len {
        let mut sums = [0.0f32; 4];
        for j in 0..4 {
            sums[j] = coeffs[0] * lines[0][i + j]
                + coeffs[1] * lines[1][i + j]
                + coeffs[2] * lines[2][i + j]
                + coeffs[3] * lines[3][i + j];
        }
        let alpha = clampf(sums[3]);
        for j in 0..3 {
            if alpha != 0.0 && alpha != 1.0 {
                sums[j] /= alpha;
                sums[j] = clampf(sums[j]);
            }
            out[i + j] = tables.linear_to_srgb(sums[j]);
        }
        out[i + 3] = f2i(alpha * 255.0);
        i += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let mut sum = [[0.0f32; 4]; 3];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            for k in 0..3 {
                add_sample_to_sum(tables.s2l[input[in_idx + k] as usize], cx, &mut sum[k]);
            }
            in_idx += 3;
            cx_idx += 4;
        }

        for j in 0..3 {
            add_sample_to_sum(sum[j][0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_rgb(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let tables = srgb::tables();
    let mut s_idx = 0usize;
    for i in 0..sl_len {
        out[i] = tables.linear_to_srgb(sums[s_idx]);
        shift_left(&mut sums[s_idx..s_idx + 4]);
        s_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgba(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let off0 = tap;
    let off1 = (tap + 1) & 3;
    let off2 = (tap + 2) & 3;
    let off3 = (tap + 3) & 3;
    let mut sum = [[0.0f32; 4]; 4]; // R, G, B, A
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            let alpha = tables.i2f[input[in_idx + 3] as usize];
            for k in 0..3 {
                add_sample_to_sum(tables.s2l[input[in_idx + k] as usize] * alpha, cx, &mut sum[k]);
            }
            add_sample_to_sum(alpha, cx, &mut sum[3]);
            in_idx += 4;
            cx_idx += 4;
        }

        for j in 0..4 {
            let sample = sum[j][0];
            sums_y[sy_idx + off0] += sample * coeffs_y[0];
            sums_y[sy_idx + off1] += sample * coeffs_y[1];
            sums_y[sy_idx + off2] += sample * coeffs_y[2];
            sums_y[sy_idx + off3] += sample * coeffs_y[3];
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_rgba(sums: &mut [f32], width: usize, out: &mut [u8], tap: usize) {
    let tables = srgb::tables();
    let tap_off = tap;
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let alpha = clampf(sums[s_idx + 12 + tap_off]);
        for j in 0..3 {
            let mut val = sums[s_idx + j * 4 + tap_off];
            if alpha != 0.0 {
                val /= alpha;
            }
            out[o_idx + j] = tables.linear_to_srgb(clampf(val));
            sums[s_idx + j * 4 + tap_off] = 0.0;
        }
        out[o_idx + 3] = f2i(alpha * 255.0);
        sums[s_idx + 12 + tap_off] = 0.0;
        s_idx += 16;
        o_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgbx(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 4]; // R, G, B, X
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        for j in 0..3 {
            push_f(&mut smp[j], tables.s2l[input[in_base + j] as usize]);
        }
        push_f(&mut smp[3], 1.0);
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..4 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_rgbx(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let mut i = 0;
    while i < len {
        let mut sums = [0.0f32; 4];
        for j in 0..4 {
            sums[j] = coeffs[0] * lines[0][i + j]
                + coeffs[1] * lines[1][i + j]
                + coeffs[2] * lines[2][i + j]
                + coeffs[3] * lines[3][i + j];
        }
        for j in 0..3 {
            out[i + j] = tables.linear_to_srgb(clampf(sums[j]));
        }
        out[i + 3] = 255;
        i += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgbx(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let off0 = tap;
    let off1 = (tap + 1) & 3;
    let off2 = (tap + 2) & 3;
    let off3 = (tap + 3) & 3;
    let mut sum = [[0.0f32; 4]; 4]; // R, G, B, X
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            let px = u32::from_le_bytes([input[in_idx], input[in_idx+1], input[in_idx+2], input[in_idx+3]]);
            add_sample_to_sum(tables.s2l[(px & 0xFF) as usize], cx, &mut sum[0]);
            add_sample_to_sum(tables.s2l[((px >> 8) & 0xFF) as usize], cx, &mut sum[1]);
            add_sample_to_sum(tables.s2l[((px >> 16) & 0xFF) as usize], cx, &mut sum[2]);
            add_sample_to_sum(1.0, cx, &mut sum[3]);
            in_idx += 4;
            cx_idx += 4;
        }

        for j in 0..4 {
            let sample = sum[j][0];
            sums_y[sy_idx + off0] += sample * coeffs_y[0];
            sums_y[sy_idx + off1] += sample * coeffs_y[1];
            sums_y[sy_idx + off2] += sample * coeffs_y[2];
            sums_y[sy_idx + off3] += sample * coeffs_y[3];
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_rgbx(sums: &mut [f32], width: usize, out: &mut [u8], tap: usize) {
    let tables = srgb::tables();
    let tap_off = tap;
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        for j in 0..3 {
            out[o_idx + j] = tables.linear_to_srgb(clampf(sums[s_idx + j * 4 + tap_off]));
            sums[s_idx + j * 4 + tap_off] = 0.0;
        }
        out[o_idx + 3] = 255;
        sums[s_idx + 12 + tap_off] = 0.0;
        s_idx += 16;
        o_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_cmyk(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let mut smp = [[0.0f32; 4]; 4]; // C, M, Y, K
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        for j in 0..4 {
            push_f(&mut smp[j], input[in_base + j] as f32 / 255.0);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..4 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_cmyk(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let mut sum = [[0.0f32; 4]; 4]; // C, M, Y, K
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            let px = u32::from_le_bytes([input[in_idx], input[in_idx+1], input[in_idx+2], input[in_idx+3]]);
            add_sample_to_sum(tables.i2f[(px & 0xFF) as usize], cx, &mut sum[0]);
            add_sample_to_sum(tables.i2f[((px >> 8) & 0xFF) as usize], cx, &mut sum[1]);
            add_sample_to_sum(tables.i2f[((px >> 16) & 0xFF) as usize], cx, &mut sum[2]);
            add_sample_to_sum(tables.i2f[(px >> 24) as usize], cx, &mut sum[3]);
            in_idx += 4;
            cx_idx += 4;
        }

        for j in 0..4 {
            add_sample_to_sum(sum[j][0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

// --- NOGAMMA scalar functions ---
// No SSE2 paths yet, so these are always compiled.

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgb_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 3];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;
        for j in 0..3 {
            push_f(&mut smp[j], tables.i2f[input[in_base + j] as usize]);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..3 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgba_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 4];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha = tables.i2f[input[in_base + 3] as usize];
        push_f(&mut smp[3], alpha);
        let alpha_val = smp[3][3];
        for j in 0..3 {
            push_f(&mut smp[j], alpha_val * tables.i2f[input[in_base + j] as usize]);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..4 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn xscale_up_rgbx_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 4];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        for j in 0..3 {
            push_f(&mut smp[j], tables.i2f[input[in_base + j] as usize]);
        }
        push_f(&mut smp[3], 1.0);
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..4 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_rgba_nogamma(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let mut i = 0;
    while i < len {
        let mut sums = [0.0f32; 4];
        for j in 0..4 {
            sums[j] = coeffs[0] * lines[0][i + j]
                + coeffs[1] * lines[1][i + j]
                + coeffs[2] * lines[2][i + j]
                + coeffs[3] * lines[3][i + j];
        }
        let alpha = clampf(sums[3]);
        for j in 0..3 {
            if alpha != 0.0 && alpha != 1.0 {
                sums[j] /= alpha;
                sums[j] = clampf(sums[j]);
            }
            out[i + j] = f2i(clampf(sums[j]) * 255.0);
        }
        out[i + 3] = f2i(alpha * 255.0);
        i += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_up_rgbx_nogamma(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let mut i = 0;
    while i < len {
        let mut sums = [0.0f32; 4];
        for j in 0..4 {
            sums[j] = coeffs[0] * lines[0][i + j]
                + coeffs[1] * lines[1][i + j]
                + coeffs[2] * lines[2][i + j]
                + coeffs[3] * lines[3][i + j];
        }
        for j in 0..3 {
            out[i + j] = f2i(clampf(sums[j]) * 255.0);
        }
        out[i + 3] = 255;
        i += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgb_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let mut sum = [[0.0f32; 4]; 3];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            for k in 0..3 {
                add_sample_to_sum(tables.i2f[input[in_idx + k] as usize], cx, &mut sum[k]);
            }
            in_idx += 3;
            cx_idx += 4;
        }

        for j in 0..3 {
            add_sample_to_sum(sum[j][0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgba_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let off0 = tap;
    let off1 = (tap + 1) & 3;
    let off2 = (tap + 2) & 3;
    let off3 = (tap + 3) & 3;
    let mut sum = [[0.0f32; 4]; 4];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            let alpha = tables.i2f[input[in_idx + 3] as usize];
            for k in 0..3 {
                add_sample_to_sum(tables.i2f[input[in_idx + k] as usize] * alpha, cx, &mut sum[k]);
            }
            add_sample_to_sum(alpha, cx, &mut sum[3]);
            in_idx += 4;
            cx_idx += 4;
        }

        for j in 0..4 {
            let sample = sum[j][0];
            sums_y[sy_idx + off0] += sample * coeffs_y[0];
            sums_y[sy_idx + off1] += sample * coeffs_y[1];
            sums_y[sy_idx + off2] += sample * coeffs_y[2];
            sums_y[sy_idx + off3] += sample * coeffs_y[3];
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn scale_down_rgbx_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let off0 = tap;
    let off1 = (tap + 1) & 3;
    let off2 = (tap + 2) & 3;
    let off3 = (tap + 3) & 3;
    let mut sum = [[0.0f32; 4]; 4];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            for k in 0..3 {
                add_sample_to_sum(tables.i2f[input[in_idx + k] as usize], cx, &mut sum[k]);
            }
            add_sample_to_sum(1.0, cx, &mut sum[3]);
            in_idx += 4;
            cx_idx += 4;
        }

        for j in 0..4 {
            let sample = sum[j][0];
            sums_y[sy_idx + off0] += sample * coeffs_y[0];
            sums_y[sy_idx + off1] += sample * coeffs_y[1];
            sums_y[sy_idx + off2] += sample * coeffs_y[2];
            sums_y[sy_idx + off3] += sample * coeffs_y[3];
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_rgba_nogamma(sums: &mut [f32], width: usize, out: &mut [u8], tap: usize) {
    let tap_off = tap;
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let alpha = clampf(sums[s_idx + 12 + tap_off]);
        for j in 0..3 {
            let mut val = sums[s_idx + j * 4 + tap_off];
            if alpha != 0.0 {
                val /= alpha;
            }
            out[o_idx + j] = f2i(clampf(val) * 255.0);
            sums[s_idx + j * 4 + tap_off] = 0.0;
        }
        out[o_idx + 3] = f2i(alpha * 255.0);
        sums[s_idx + 12 + tap_off] = 0.0;
        s_idx += 16;
        o_idx += 4;
    }
}

#[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
fn yscale_out_rgbx_nogamma(sums: &mut [f32], width: usize, out: &mut [u8], tap: usize) {
    let tap_off = tap;
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        for j in 0..3 {
            out[o_idx + j] = f2i(clampf(sums[s_idx + j * 4 + tap_off]) * 255.0);
            sums[s_idx + j * 4 + tap_off] = 0.0;
        }
        out[o_idx + 3] = 255;
        sums[s_idx + 12 + tap_off] = 0.0;
        s_idx += 16;
        o_idx += 4;
    }
}

impl OilScale {
    /// Create a new scaler for the given input/output dimensions and color space.
    pub fn new(
        in_width: u32,
        in_height: u32,
        out_width: u32,
        out_height: u32,
        cs: ColorSpace,
    ) -> Result<Self, Error> {
        if in_height < 1
            || out_height < 1
            || in_width < 1
            || out_width < 1
            || in_height > MAX_DIMENSION
            || out_height > MAX_DIMENSION
            || in_width > MAX_DIMENSION
            || out_width > MAX_DIMENSION
        {
            return Err(Error::InvalidArgument);
        }

        // Only allow upscaling if both dimensions are being upscaled
        if (out_height > in_height) != (out_width > in_width) {
            return Err(Error::InvalidArgument);
        }

        // Ensure tables are initialized
        srgb::tables();

        let is_upscale = out_width > in_width;

        let mut os = OilScale {
            in_height,
            out_height,
            in_width,
            out_width,
            cs,
            in_pos: 0,
            out_pos: 0,
            coeffs_y: Vec::new(),
            coeffs_x: Vec::new(),
            borders_x: Vec::new(),
            borders_y: Vec::new(),
            sums_y: Vec::new(),
            sums_y_tap: 0,
            rb: Vec::new(),
            tmp_coeffs: Vec::new(),
            is_upscale,
        };

        if is_upscale {
            os.upscale_init();
        } else {
            os.downscale_init();
        }

        Ok(os)
    }

    /// Returns the input image height in pixels.
    pub fn input_height(&self) -> u32 {
        self.in_height
    }

    /// Returns the output image height in pixels.
    pub fn output_height(&self) -> u32 {
        self.out_height
    }

    /// Returns the input image width in pixels.
    pub fn input_width(&self) -> u32 {
        self.in_width
    }

    /// Returns the output image width in pixels.
    pub fn output_width(&self) -> u32 {
        self.out_width
    }

    /// Returns the color space used by this scaler.
    pub fn color_space(&self) -> ColorSpace {
        self.cs
    }

    /// Returns `true` if this scaler is upscaling (output dimensions larger
    /// than input dimensions).
    pub fn is_upscale(&self) -> bool {
        self.is_upscale
    }

    fn upscale_init(&mut self) {
        let cmp = self.cs.components();
        let coeffs_x_len = TAPS * self.out_width.max(self.in_width) as usize;
        let borders_x_len = self.in_width as usize;
        let coeffs_y_len = TAPS * self.out_height.max(self.in_height) as usize;
        let borders_y_len = self.in_height as usize;
        let rb_len = self.out_width as usize * cmp * TAPS;

        self.coeffs_x = vec![0.0; coeffs_x_len];
        self.borders_x = vec![0; borders_x_len];
        self.coeffs_y = vec![0.0; coeffs_y_len];
        self.borders_y = vec![0; borders_y_len];
        self.rb = vec![0.0; rb_len];

        kernel::scale_up_coeffs(
            self.in_width,
            self.out_width,
            &mut self.coeffs_x,
            &mut self.borders_x,
        );
        kernel::scale_up_coeffs(
            self.in_height,
            self.out_height,
            &mut self.coeffs_y,
            &mut self.borders_y,
        );
    }

    fn downscale_init(&mut self) {
        let cmp = self.cs.components();
        let taps_x = kernel::calc_taps(self.in_width, self.out_width);
        let taps_y = kernel::calc_taps(self.in_height, self.out_height);

        let coeffs_x_len = TAPS * self.in_width.max(self.out_width) as usize;
        let borders_x_len = self.out_width as usize;
        let coeffs_y_len = TAPS * self.in_height.max(self.out_height) as usize;
        let borders_y_len = self.out_height as usize;
        let sums_len = self.out_width as usize * cmp * TAPS;
        let tmp_len = taps_x.max(taps_y);

        self.coeffs_x = vec![0.0; coeffs_x_len];
        self.borders_x = vec![0; borders_x_len];
        self.coeffs_y = vec![0.0; coeffs_y_len];
        self.borders_y = vec![0; borders_y_len];
        self.sums_y = vec![0.0; sums_len];
        self.tmp_coeffs = vec![0.0; tmp_len];

        kernel::scale_down_coeffs(
            self.in_width,
            self.out_width,
            &mut self.coeffs_x,
            &mut self.borders_x,
            &mut self.tmp_coeffs,
        );
        kernel::scale_down_coeffs(
            self.in_height,
            self.out_height,
            &mut self.coeffs_y,
            &mut self.borders_y,
            &mut self.tmp_coeffs,
        );
    }

    /// Return the number of input scanlines needed before the next output
    /// scanline can be produced.
    pub fn slots(&self) -> usize {
        if !self.is_upscale {
            self.borders_y[self.out_pos as usize] as usize
        } else if self.in_pos > 0 {
            if self.borders_y[self.in_pos as usize - 1] == 0 {
                1
            } else {
                0
            }
        } else if self.borders_y[0] == 0 {
            2
        } else {
            1
        }
    }

    /// Ingest one input scanline.
    ///
    /// Returns `Err(Error::InvalidArgument)` if an output scanline is ready but
    /// has not been consumed (i.e. `slots()` returns 0). Feeding input while an
    /// output line is pending would corrupt internal state.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` is less than `input_width() * color_space().components()`,
    /// or if called more times than the input height without a [`reset`](Self::reset).
    pub fn push_scanline(&mut self, input: &[u8]) -> Result<(), Error> {
        if self.slots() == 0 {
            return Err(Error::InvalidArgument);
        }
        if self.is_upscale {
            self.up_scale_in(input);
        } else {
            self.down_scale_in(input);
        }
        Ok(())
    }

    /// Produce the next scaled output scanline.
    ///
    /// Returns `Err(Error::InvalidArgument)` if not enough input scanlines have
    /// been fed yet (i.e. `slots()` is not 0).
    ///
    /// # Panics
    ///
    /// Panics if `output.len()` is less than `output_width() * color_space().components()`,
    /// or if called more than `output_height()` times without a [`reset`](Self::reset).
    pub fn read_scanline(&mut self, output: &mut [u8]) -> Result<(), Error> {
        if self.slots() != 0 {
            return Err(Error::InvalidArgument);
        }
        if self.is_upscale {
            self.up_scale_out(output);
        } else {
            self.down_scale_out(output);
        }
        self.out_pos += 1;
        Ok(())
    }

    /// Skip the next output scanline without producing it.
    ///
    /// This advances internal state so that input feeding can continue, but
    /// does not write any pixel data. Useful when a caller wants to discard
    /// certain output rows without the cost of computing them.
    ///
    /// Returns `Err(Error::InvalidArgument)` if not enough input scanlines have
    /// been fed yet (i.e. `slots()` is not 0).
    ///
    /// # Panics
    ///
    /// Panics if called more than `output_height()` times without a
    /// [`reset`](Self::reset).
    pub fn discard_output_scanline(&mut self) -> Result<(), Error> {
        if self.slots() != 0 {
            return Err(Error::InvalidArgument);
        }
        if self.is_upscale {
            self.borders_y[self.in_pos as usize - 1] -= 1;
        } else {
            // Use yscale_out to shift the sums_y accumulators, discarding
            // the output pixels. This avoids needing layout-specific shift
            // logic for each colorspace's sums_y memory layout.
            let cmp = self.cs.components();
            let sl_len = self.out_width as usize * cmp;
            let mut tmp = vec![0u8; sl_len];
            self.down_scale_out(&mut tmp);
        }
        self.out_pos += 1;
        Ok(())
    }

    /// Reset the scaler so it can process another image of the same dimensions.
    ///
    /// This recalculates internal state that is consumed during processing
    /// (border counters and accumulator buffers).
    pub fn reset(&mut self) {
        self.in_pos = 0;
        self.out_pos = 0;
        self.sums_y_tap = 0;
        if self.is_upscale {
            self.upscale_init();
        } else {
            self.downscale_init();
        }
    }

    fn get_rb_line(&self, line: u32) -> usize {
        let sl_len = self.cs.components() * self.out_width as usize;
        line as usize * sl_len
    }

    fn up_scale_in(&mut self, input: &[u8]) {
        let rb_offset = self.get_rb_line(self.in_pos % 4);
        let sl_len = self.cs.components() * self.out_width as usize;

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgb(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgb(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgb(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::RGBA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgba(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgba(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgba(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::RGBX => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgbx(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgbx(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgbx(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::G => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_g(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_g(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_g(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::GA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_ga(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_ga(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_ga(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::CMYK => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_cmyk(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_cmyk(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_cmyk(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::RgbNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgb_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgb_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgb_nogamma(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::RgbaNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgba_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgba_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgba_nogamma(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            ColorSpace::RgbxNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::xscale_up_rgbx_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::xscale_up_rgbx_nogamma(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                xscale_up_rgbx_nogamma(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
        }

        self.in_pos += 1;
    }

    fn up_scale_out(&mut self, output: &mut [u8]) {
        let cmp = self.cs.components();
        let sl_len = cmp * self.out_width as usize;

        let offsets: [usize; 4] = [
            self.get_rb_line(self.in_pos % 4),
            self.get_rb_line((self.in_pos + 1) % 4),
            self.get_rb_line((self.in_pos + 2) % 4),
            self.get_rb_line((self.in_pos + 3) % 4),
        ];

        let lines: [&[f32]; 4] = [
            &self.rb[offsets[0]..offsets[0] + sl_len],
            &self.rb[offsets[1]..offsets[1] + sl_len],
            &self.rb[offsets[2]..offsets[2] + sl_len],
            &self.rb[offsets[3]..offsets[3] + sl_len],
        ];

        let coeff_start = self.out_pos as usize * 4;
        let coeffs = &self.coeffs_y[coeff_start..coeff_start + 4];

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_rgb(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_rgb(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_rgb(lines, sl_len, coeffs, output);
            }
            ColorSpace::RGBA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_rgba(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_rgba(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_rgba(lines, sl_len, coeffs, output);
            }
            ColorSpace::RGBX => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_rgbx(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_rgbx(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_rgbx(lines, sl_len, coeffs, output);
            }
            ColorSpace::G | ColorSpace::CMYK | ColorSpace::RgbNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_g(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_g(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_g(lines, sl_len, coeffs, output);
            }
            ColorSpace::GA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_ga(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_ga(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_ga(lines, sl_len, coeffs, output);
            }
            ColorSpace::RgbaNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_rgba_nogamma(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_rgba_nogamma(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_rgba_nogamma(lines, sl_len, coeffs, output);
            }
            ColorSpace::RgbxNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_up_rgbx_nogamma(lines, sl_len, coeffs, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_up_rgbx_nogamma(lines, sl_len, coeffs, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_up_rgbx_nogamma(lines, sl_len, coeffs, output);
            }
        }

        self.borders_y[self.in_pos as usize - 1] -= 1;
    }

    fn down_scale_in(&mut self, input: &[u8]) {
        let coeffs_y_start = self.in_pos as usize * 4;
        let coeffs_y = [
            self.coeffs_y[coeffs_y_start],
            self.coeffs_y[coeffs_y_start + 1],
            self.coeffs_y[coeffs_y_start + 2],
            self.coeffs_y[coeffs_y_start + 3],
        ];

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgb(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgb(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgb(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            ColorSpace::RGBA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgba(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                        self.sums_y_tap,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgba(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgba(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                    self.sums_y_tap,
                );
            }
            ColorSpace::RGBX => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgbx(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                        self.sums_y_tap,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgbx(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgbx(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                    self.sums_y_tap,
                );
            }
            ColorSpace::G => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_g(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_g(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_g(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            ColorSpace::GA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_ga(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_ga(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_ga(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            ColorSpace::CMYK => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_cmyk(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_cmyk(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_cmyk(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            ColorSpace::RgbNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgb_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgb_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgb_nogamma(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            ColorSpace::RgbaNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgba_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                        self.sums_y_tap,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgba_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgba_nogamma(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                    self.sums_y_tap,
                );
            }
            ColorSpace::RgbxNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::scale_down_rgbx_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                        self.sums_y_tap,
                    );
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::scale_down_rgbx_nogamma(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                scale_down_rgbx_nogamma(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                    self.sums_y_tap,
                );
            }
        }

        self.borders_y[self.out_pos as usize] -= 1;
        self.in_pos += 1;
    }
}

/// Adjust output dimensions to maintain the input aspect ratio, fitting within
/// the given bounding box. Returns the adjusted `(width, height)`.
pub fn fix_ratio(
    src_width: u32,
    src_height: u32,
    max_width: u32,
    max_height: u32,
) -> Result<(u32, u32), Error> {
    if src_width < 1 || src_height < 1 || max_width < 1 || max_height < 1 {
        return Err(Error::InvalidArgument);
    }

    let width_ratio = max_width as f64 / src_width as f64;
    let height_ratio = max_height as f64 / src_height as f64;

    if width_ratio < height_ratio {
        let tmp = (width_ratio * src_height as f64).round();
        let h = if tmp < 1.0 { 1 } else { tmp as u32 };
        Ok((max_width, h))
    } else {
        let tmp = (height_ratio * src_width as f64).round();
        let w = if tmp < 1.0 { 1 } else { tmp as u32 };
        Ok((w, max_height))
    }
}

impl OilScale {
    fn down_scale_out(&mut self, output: &mut [u8]) {
        let cmp = self.cs.components();
        let sl_len = self.out_width as usize * cmp;
        let tap = self.sums_y_tap;

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_rgb(&mut self.sums_y, sl_len, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_rgb(&mut self.sums_y, sl_len, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_rgb(&mut self.sums_y, sl_len, output);
            }
            ColorSpace::RGBA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_rgba(&mut self.sums_y, self.out_width, output, tap);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_rgba(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_rgba(&mut self.sums_y, self.out_width as usize, output, tap);
            }
            ColorSpace::RGBX => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_rgbx(&mut self.sums_y, self.out_width, output, tap);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_rgbx(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_rgbx(&mut self.sums_y, self.out_width as usize, output, tap);
            }
            ColorSpace::G | ColorSpace::CMYK | ColorSpace::RgbNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_g(&mut self.sums_y, sl_len, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_g(&mut self.sums_y, sl_len, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_g(&mut self.sums_y, sl_len, output);
            }
            ColorSpace::GA => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_ga(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_ga(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_ga(&mut self.sums_y, self.out_width as usize, output);
            }
            ColorSpace::RgbaNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_rgba_nogamma(&mut self.sums_y, self.out_width, output, tap);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_rgba_nogamma(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_rgba_nogamma(&mut self.sums_y, self.out_width as usize, output, tap);
            }
            ColorSpace::RgbxNoGamma => {
                #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
                unsafe {
                    sse2::yscale_out_rgbx_nogamma(&mut self.sums_y, self.out_width, output, tap);
                }
                #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
                unsafe {
                    neon::yscale_out_rgbx_nogamma(&mut self.sums_y, self.out_width, output);
                }
                #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), feature = "force-scalar"))]
                yscale_out_rgbx_nogamma(&mut self.sums_y, self.out_width as usize, output, tap);
            }
        }

        self.sums_y_tap = (self.sums_y_tap + 1) & 3;
    }
}
