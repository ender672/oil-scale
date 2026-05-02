const TAPS: usize = 4;

/// Catmull-Rom interpolation kernel.
#[inline]
pub fn catrom(x: f32) -> f32 {
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else {
        (((5.0 - x) * x - 8.0) * x + 4.0) / 2.0
    }
}

/// Calculate the number of taps needed for resampling.
/// Upscale always uses 4 taps. Downscale widens the kernel to prevent aliasing.
pub fn calc_taps(dim_in: u32, dim_out: u32) -> usize {
    if dim_out > dim_in {
        TAPS
    } else {
        let tmp = (TAPS as u32 * dim_in / dim_out) as usize;
        tmp - (tmp & 1)
    }
}

/// Calculate filter coefficients for a given fractional offset.
/// Coefficients are normalized to sum to 1.0.
pub fn calc_coeffs(coeffs: &mut [f32], tx: f32, taps: usize, ltrim: usize, rtrim: usize) {
    let tap_mult = taps as f32 / TAPS as f32;
    let mut pos = 1.0 - tx - (taps / 2) as f32 + ltrim as f32;
    let mut fudge = 0.0f32;

    for c in &mut coeffs[ltrim..(taps - rtrim)] {
        let tmp = catrom(pos.abs() / tap_mult) / tap_mult;
        fudge += tmp;
        *c = tmp;
        pos += 1.0;
    }

    let fudge = 1.0 / fudge;
    for c in &mut coeffs[ltrim..(taps - rtrim)] {
        *c *= fudge;
    }
}

/// Pre-calculate coefficients and border counters for upscaling.
///
/// coeff_buf: 4 coefficients per output sample (4 * out_dim total).
/// border_buf: number of output samples to produce for each input sample (in_dim entries).
pub fn scale_up_coeffs(
    in_dim: u32,
    out_dim: u32,
    coeff_buf: &mut [f32],
    border_buf: &mut [i32],
) {
    let max_pos = in_dim as i32 - 1;
    let mut coeff_offset = 0usize;

    let step = in_dim as f64 / out_dim as f64;
    let mut pos_d = 0.5 * step - 0.5;

    for _ in 0..out_dim {
        let smp_i = if pos_d < 0.0 { -1 } else { pos_d as i32 };
        let tx = (pos_d - smp_i as f64) as f32;
        let start = smp_i - 1;
        let end = smp_i + 2;

        let safe_end = end.min(max_pos) as usize;

        let mut ltrim = 0usize;
        let mut rtrim = 0usize;
        if start < 0 {
            ltrim = (-start) as usize;
        }
        if end > max_pos {
            rtrim = (end - max_pos) as usize;
        }

        border_buf[safe_end] += 1;

        calc_coeffs(
            &mut coeff_buf[coeff_offset + rtrim..],
            tx,
            4,
            ltrim,
            rtrim,
        );

        coeff_offset += 4;
        pos_d += step;
    }
}

/// Pre-calculate coefficients and border counters for downscaling.
///
/// coeff_buf: 4 coefficients per input sample (4 * in_dim total).
/// border_buf: number of input samples to process before next output is finished (out_dim entries).
/// tmp_coeffs: temporary buffer of size >= taps.
pub fn scale_down_coeffs(
    in_dim: u32,
    out_dim: u32,
    coeff_buf: &mut [f32],
    border_buf: &mut [i32],
    tmp_coeffs: &mut [f32],
) {
    let taps = calc_taps(in_dim, out_dim);
    let mut ends = [-1i32; 4];

    let tap_mult = in_dim as f64 / out_dim as f64;
    let mut center = 0.5 * tap_mult - 0.5;

    for i in 0..out_dim as usize {
        let smp_i = if center < 0.0 { -1 } else { center as i32 };
        let tx = (center - smp_i as f64) as f32;

        let smp_start = smp_i - (taps as i32 / 2 - 1);
        let mut smp_end = smp_i + taps as i32 / 2;
        if smp_end >= in_dim as i32 {
            smp_end = in_dim as i32 - 1;
        }
        ends[i % 4] = smp_end;
        border_buf[i] = smp_end - ends[(i + 3) % 4];

        let mut ltrim = 0usize;
        if smp_start < 0 {
            ltrim = (-smp_start) as usize;
        }
        let rtrim = (smp_start + taps as i32 - 1 - smp_end) as usize;

        // Zero tmp_coeffs before use
        for c in tmp_coeffs.iter_mut().take(taps) {
            *c = 0.0;
        }
        calc_coeffs(tmp_coeffs, tx, taps, ltrim, rtrim);

        for (j, &coeff) in tmp_coeffs.iter().enumerate().take(taps - rtrim).skip(ltrim) {
            let pos = (smp_start + j as i32) as usize;

            let offset = if pos as i32 > ends[(i + 3) % 4] {
                0
            } else if pos as i32 > ends[(i + 2) % 4] {
                1
            } else if pos as i32 > ends[(i + 1) % 4] {
                2
            } else {
                3
            };

            coeff_buf[pos * 4 + offset] = coeff;
        }

        center += tap_mult;
    }
}
