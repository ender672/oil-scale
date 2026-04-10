use std::arch::x86_64::*;

use crate::srgb;

/// Equivalent to C's mm_shuffle(z, y, x, w).
const fn mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

/// AVX2 downscale for G: horizontal x-filtering + 256-bit y-accumulation.
/// Processes 2 output pixels at a time using 256-bit AVX2 for vertical accumulation.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_g(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy128 = _mm_loadu_ps(coeffs_y.as_ptr());
    let cy256 = _mm256_set_m128(cy128, cy128);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;
    let mut sum = _mm_setzero_ps();

    let mut i = 0u32;

    // Process pairs of output pixels with 256-bit y-accumulation
    while i + 1 < out_width {
        let border0 = *border_ptr.add(i as usize);
        for _j in 0..border0 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
        }
        let result_lo = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        let border1 = *border_ptr.add(i as usize + 1);
        for _j in 0..border1 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
        }
        let result_hi = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        let mut sy256 = _mm256_loadu_ps(sy_ptr.add(sy_idx));
        let sample256 = _mm256_set_m128(result_hi, result_lo);
        sy256 = _mm256_add_ps(_mm256_mul_ps(cy256, sample256), sy256);
        _mm256_storeu_ps(sy_ptr.add(sy_idx), sy256);
        sy_idx += 8;
        i += 2;
    }

    // Remaining single pixel
    if i < out_width {
        let cy = _mm256_castps256_ps128(cy256);
        let border = *border_ptr.add(i as usize);
        for _j in 0..border {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
        }
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
    }
}

/// AVX2 heavy downscale for G (when in_width >= out_width * 2).
/// Uses 4x loop unrolling in the inner x-loop with independent accumulators,
/// plus 256-bit AVX2 for vertical accumulation.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_g_heavy(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy128 = _mm_loadu_ps(coeffs_y.as_ptr());
    let cy256 = _mm256_set_m128(cy128, cy128);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;
    let mut sum = _mm_setzero_ps();

    let mut i = 0u32;

    // Process pairs of output pixels
    while i + 1 < out_width {
        // First output pixel
        let border0 = *border_ptr.add(i as usize);
        let mut sum2 = _mm_setzero_ps();
        let mut sum3 = _mm_setzero_ps();
        let mut sum4 = _mm_setzero_ps();

        let mut j = 0;
        while j + 3 < border0 {
            let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx0, s0), sum);

            let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
            let s1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
            sum2 = _mm_add_ps(_mm_mul_ps(cx1, s1), sum2);

            let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
            let s2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
            sum3 = _mm_add_ps(_mm_mul_ps(cx2, s2), sum3);

            let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
            let s3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
            sum4 = _mm_add_ps(_mm_mul_ps(cx3, s3), sum4);

            in_idx += 4;
            cx_idx += 16;
            j += 4;
        }
        while j < border0 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
            j += 1;
        }
        sum = _mm_add_ps(_mm_add_ps(sum, sum2), _mm_add_ps(sum3, sum4));
        let result_lo = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        // Second output pixel
        let border1 = *border_ptr.add(i as usize + 1);
        sum2 = _mm_setzero_ps();
        sum3 = _mm_setzero_ps();
        sum4 = _mm_setzero_ps();

        j = 0;
        while j + 3 < border1 {
            let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx0, s0), sum);

            let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
            let s1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
            sum2 = _mm_add_ps(_mm_mul_ps(cx1, s1), sum2);

            let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
            let s2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
            sum3 = _mm_add_ps(_mm_mul_ps(cx2, s2), sum3);

            let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
            let s3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
            sum4 = _mm_add_ps(_mm_mul_ps(cx3, s3), sum4);

            in_idx += 4;
            cx_idx += 16;
            j += 4;
        }
        while j < border1 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
            j += 1;
        }
        sum = _mm_add_ps(_mm_add_ps(sum, sum2), _mm_add_ps(sum3, sum4));
        let result_hi = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        let mut sy256 = _mm256_loadu_ps(sy_ptr.add(sy_idx));
        let sample256 = _mm256_set_m128(result_hi, result_lo);
        sy256 = _mm256_add_ps(_mm256_mul_ps(cy256, sample256), sy256);
        _mm256_storeu_ps(sy_ptr.add(sy_idx), sy256);
        sy_idx += 8;
        i += 2;
    }

    // Remaining single pixel
    if i < out_width {
        let cy = _mm256_castps256_ps128(cy256);
        let border = *border_ptr.add(i as usize);
        let mut sum2 = _mm_setzero_ps();
        let mut sum3 = _mm_setzero_ps();
        let mut sum4 = _mm_setzero_ps();

        let mut j = 0;
        while j + 3 < border {
            let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx0, s0), sum);

            let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
            let s1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
            sum2 = _mm_add_ps(_mm_mul_ps(cx1, s1), sum2);

            let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
            let s2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
            sum3 = _mm_add_ps(_mm_mul_ps(cx2, s2), sum3);

            let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
            let s3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
            sum4 = _mm_add_ps(_mm_mul_ps(cx3, s3), sum4);

            in_idx += 4;
            cx_idx += 16;
            j += 4;
        }
        while j < border {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
            j += 1;
        }
        sum = _mm_add_ps(_mm_add_ps(sum, sum2), _mm_add_ps(sum3, sum4));
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
    }
}

/// AVX2 downscale for RGBX_NOGAMMA: FMA x-filtering + 256-bit y-accumulation + prefetch.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_rgbx_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();

    // Precompute 256-bit coefficient vectors ordered by physical slot
    let cy_lo;
    let cy_hi;
    {
        let mut cy_slot = [0.0f32; 4];
        for k in 0..4 {
            cy_slot[k] = coeffs_y[(k + 4 - tap) & 3];
        }
        cy_lo = _mm256_set_m128(
            _mm_set1_ps(cy_slot[1]),
            _mm_set1_ps(cy_slot[0]),
        );
        cy_hi = _mm256_set_m128(
            _mm_set1_ps(cy_slot[3]),
            _mm_set1_ps(cy_slot[2]),
        );
    }

    let mut sum_r = _mm_setzero_ps();
    let mut sum_g = _mm_setzero_ps();
    let mut sum_b = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 4 {
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_fmadd_ps(cx2, s, sum_r2);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_fmadd_ps(cx2, s, sum_g2);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_fmadd_ps(cx2, s, sum_b2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = _mm_add_ps(sum_r, sum_r2);
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_b = _mm_add_ps(sum_b, sum_b2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Vertical accumulation using 256-bit AVX2
        // Prefetch next pixel's sums_y
        _mm_prefetch(sy_ptr.add(sy_idx + 16) as *const i8, _MM_HINT_T0);

        let rg = _mm_unpacklo_ps(sum_r, sum_g);
        let bx = _mm_unpacklo_ps(sum_b, sum_b);
        let rgbx = _mm_movelh_ps(rg, bx);

        let rgbx256 = _mm256_set_m128(rgbx, rgbx);

        let mut sy = _mm256_loadu_ps(sy_ptr.add(sy_idx));
        sy = _mm256_fmadd_ps(cy_lo, rgbx256, sy);
        _mm256_storeu_ps(sy_ptr.add(sy_idx), sy);

        sy = _mm256_loadu_ps(sy_ptr.add(sy_idx + 8));
        sy = _mm256_fmadd_ps(cy_hi, rgbx256, sy);
        _mm256_storeu_ps(sy_ptr.add(sy_idx + 8), sy);

        sy_idx += 16;

        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
    }
}

/// AVX2 downscale for RGBA_NOGAMMA: horizontal x-filtering with premultiplied alpha + 256-bit FMA y-accumulation.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_rgba_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();

    // Precompute 256-bit coefficient vectors ordered by physical slot
    let cy256_lo;
    let cy256_hi;
    {
        let mut cy_phys = [0.0f32; 4];
        cy_phys[tap & 3] = coeffs_y[0];
        cy_phys[(tap + 1) & 3] = coeffs_y[1];
        cy_phys[(tap + 2) & 3] = coeffs_y[2];
        cy_phys[(tap + 3) & 3] = coeffs_y[3];
        cy256_lo = _mm256_set_m128(
            _mm_set1_ps(cy_phys[1]),
            _mm_set1_ps(cy_phys[0]),
        );
        cy256_hi = _mm256_set_m128(
            _mm_set1_ps(cy_phys[3]),
            _mm_set1_ps(cy_phys[2]),
        );
    }

    let mut sum_r = _mm_setzero_ps();
    let mut sum_g = _mm_setzero_ps();
    let mut sum_b = _mm_setzero_ps();
    let mut sum_a = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 4 {
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();
            let mut sum_a2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);
                sum_a = _mm_add_ps(cx_a, sum_a);

                let cx2_a = _mm_mul_ps(cx2, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 7) as usize)));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_r2);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_g2);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_b2);
                sum_a2 = _mm_add_ps(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);
                sum_a = _mm_add_ps(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = _mm_add_ps(sum_r, sum_r2);
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_b = _mm_add_ps(sum_b, sum_b2);
            sum_a = _mm_add_ps(sum_a, sum_a2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);
                sum_a = _mm_add_ps(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Vertical accumulation using 256-bit FMA
        let rg = _mm_unpacklo_ps(sum_r, sum_g);
        let ba = _mm_unpacklo_ps(sum_b, sum_a);
        let rgba = _mm_movelh_ps(rg, ba);

        let rgba256 = _mm256_set_m128(rgba, rgba);
        let mut sy_lo = _mm256_loadu_ps(sy_ptr.add(sy_idx));
        let mut sy_hi = _mm256_loadu_ps(sy_ptr.add(sy_idx + 8));
        sy_lo = _mm256_fmadd_ps(cy256_lo, rgba256, sy_lo);
        sy_hi = _mm256_fmadd_ps(cy256_hi, rgba256, sy_hi);
        _mm256_storeu_ps(sy_ptr.add(sy_idx), sy_lo);
        _mm256_storeu_ps(sy_ptr.add(sy_idx + 8), sy_hi);

        sy_idx += 16;

        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
        sum_a = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_a), 4));
    }
}

/// AVX2 output for downscaled RGBX_NOGAMMA.
/// Processes 4 pixels at a time for wider stores.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn yscale_out_rgbx_nogamma(sums: &mut [f32], width: u32, out: &mut [u8], tap: usize) {
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let z = _mm_setzero_si128();
    let mask = _mm_set_epi32(0, -1, -1, -1);
    let x_val = _mm_set_epi32(255, 0, 0, 0);
    let tap_off = tap * 4;

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    let mut i = 0u32;

    // Process 4 pixels at a time
    while i + 3 < width {
        let v0 = _mm_loadu_ps(s_ptr.add(s_idx + tap_off));
        let v1 = _mm_loadu_ps(s_ptr.add(s_idx + 16 + tap_off));
        let v2 = _mm_loadu_ps(s_ptr.add(s_idx + 32 + tap_off));
        let v3 = _mm_loadu_ps(s_ptr.add(s_idx + 48 + tap_off));

        let v0 = _mm_min_ps(_mm_max_ps(v0, zero), one);
        let v1 = _mm_min_ps(_mm_max_ps(v1, zero), one);
        let v2 = _mm_min_ps(_mm_max_ps(v2, zero), one);
        let v3 = _mm_min_ps(_mm_max_ps(v3, zero), one);

        let i0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v0, scale), half));
        let i1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v1, scale), half));
        let i2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v2, scale), half));
        let i3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v3, scale), half));

        let i0 = _mm_or_si128(_mm_and_si128(i0, mask), x_val);
        let i1 = _mm_or_si128(_mm_and_si128(i1, mask), x_val);
        let i2 = _mm_or_si128(_mm_and_si128(i2, mask), x_val);
        let i3 = _mm_or_si128(_mm_and_si128(i3, mask), x_val);

        let p01 = _mm_packs_epi32(i0, i1);
        let p23 = _mm_packs_epi32(i2, i3);
        let packed = _mm_packus_epi16(p01, p23);
        _mm_storeu_si128(out_ptr.add(o_idx) as *mut __m128i, packed);

        _mm_storeu_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);
        _mm_storeu_si128(s_ptr.add(s_idx + 16 + tap_off) as *mut __m128i, z);
        _mm_storeu_si128(s_ptr.add(s_idx + 32 + tap_off) as *mut __m128i, z);
        _mm_storeu_si128(s_ptr.add(s_idx + 48 + tap_off) as *mut __m128i, z);

        s_idx += 64;
        o_idx += 16;
        i += 4;
    }

    // Remaining pixels
    while i < width {
        let vals = _mm_loadu_ps(s_ptr.add(s_idx + tap_off));

        let vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
        let idx = _mm_or_si128(_mm_and_si128(idx, mask), x_val);
        let packed = _mm_packs_epi32(idx, idx);
        let packed = _mm_packus_epi16(packed, packed);
        *(out_ptr.add(o_idx) as *mut i32) = _mm_cvtsi128_si32(packed);

        _mm_storeu_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);

        s_idx += 16;
        o_idx += 4;
        i += 1;
    }
}

/// AVX2 output for downscaled RGBA_NOGAMMA.
/// Processes 4 pixels at a time with unpremultiply.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn yscale_out_rgba_nogamma(sums: &mut [f32], width: u32, out: &mut [u8], tap: usize) {
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let z = _mm_setzero_si128();
    let tap_off = tap * 4;

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    let mut i = 0u32;

    // Process 4 pixels at a time
    while i + 3 < width {
        // Pixel 1
        let vals = _mm_loadu_ps(s_ptr.add(s_idx + tap_off));
        let alpha_v = _mm_shuffle_ps(vals, vals, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let mut rgb_vals = vals;
        if _mm_cvtss_f32(alpha_v) != 0.0 {
            rgb_vals = _mm_mul_ps(rgb_vals, _mm_rcp_ps(alpha_v));
        }
        rgb_vals = _mm_min_ps(_mm_max_ps(rgb_vals, zero), one);
        let hi = _mm_shuffle_ps(rgb_vals, alpha_v, mm_shuffle(0, 0, 2, 2));
        let rgb_vals = _mm_shuffle_ps(rgb_vals, hi, mm_shuffle(2, 0, 1, 0));
        let idx0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rgb_vals, scale), half));
        _mm_storeu_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);

        // Pixel 2
        let vals2 = _mm_loadu_ps(s_ptr.add(s_idx + 16 + tap_off));
        let alpha_v2 = _mm_shuffle_ps(vals2, vals2, mm_shuffle(3, 3, 3, 3));
        let alpha_v2 = _mm_min_ps(_mm_max_ps(alpha_v2, zero), one);
        let mut rgb_vals2 = vals2;
        if _mm_cvtss_f32(alpha_v2) != 0.0 {
            rgb_vals2 = _mm_mul_ps(rgb_vals2, _mm_rcp_ps(alpha_v2));
        }
        rgb_vals2 = _mm_min_ps(_mm_max_ps(rgb_vals2, zero), one);
        let hi2 = _mm_shuffle_ps(rgb_vals2, alpha_v2, mm_shuffle(0, 0, 2, 2));
        let rgb_vals2 = _mm_shuffle_ps(rgb_vals2, hi2, mm_shuffle(2, 0, 1, 0));
        let idx1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rgb_vals2, scale), half));
        _mm_storeu_si128(s_ptr.add(s_idx + 16 + tap_off) as *mut __m128i, z);

        let packed01 = _mm_packs_epi32(idx0, idx1);

        // Pixel 3
        let vals3 = _mm_loadu_ps(s_ptr.add(s_idx + 32 + tap_off));
        let alpha_v3 = _mm_shuffle_ps(vals3, vals3, mm_shuffle(3, 3, 3, 3));
        let alpha_v3 = _mm_min_ps(_mm_max_ps(alpha_v3, zero), one);
        let mut rgb_vals3 = vals3;
        if _mm_cvtss_f32(alpha_v3) != 0.0 {
            rgb_vals3 = _mm_mul_ps(rgb_vals3, _mm_rcp_ps(alpha_v3));
        }
        rgb_vals3 = _mm_min_ps(_mm_max_ps(rgb_vals3, zero), one);
        let hi3 = _mm_shuffle_ps(rgb_vals3, alpha_v3, mm_shuffle(0, 0, 2, 2));
        let rgb_vals3 = _mm_shuffle_ps(rgb_vals3, hi3, mm_shuffle(2, 0, 1, 0));
        let idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rgb_vals3, scale), half));
        _mm_storeu_si128(s_ptr.add(s_idx + 32 + tap_off) as *mut __m128i, z);

        // Pixel 4
        let vals4 = _mm_loadu_ps(s_ptr.add(s_idx + 48 + tap_off));
        let alpha_v4 = _mm_shuffle_ps(vals4, vals4, mm_shuffle(3, 3, 3, 3));
        let alpha_v4 = _mm_min_ps(_mm_max_ps(alpha_v4, zero), one);
        let mut rgb_vals4 = vals4;
        if _mm_cvtss_f32(alpha_v4) != 0.0 {
            rgb_vals4 = _mm_mul_ps(rgb_vals4, _mm_rcp_ps(alpha_v4));
        }
        rgb_vals4 = _mm_min_ps(_mm_max_ps(rgb_vals4, zero), one);
        let hi4 = _mm_shuffle_ps(rgb_vals4, alpha_v4, mm_shuffle(0, 0, 2, 2));
        let rgb_vals4 = _mm_shuffle_ps(rgb_vals4, hi4, mm_shuffle(2, 0, 1, 0));
        let idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rgb_vals4, scale), half));
        _mm_storeu_si128(s_ptr.add(s_idx + 48 + tap_off) as *mut __m128i, z);

        let packed23 = _mm_packs_epi32(idx2, idx3);
        let packed = _mm_packus_epi16(packed01, packed23);
        _mm_storeu_si128(out_ptr.add(o_idx) as *mut __m128i, packed);

        s_idx += 64;
        o_idx += 16;
        i += 4;
    }

    // Remaining pixels
    while i < width {
        let vals = _mm_loadu_ps(s_ptr.add(s_idx + tap_off));

        let alpha_v = _mm_shuffle_ps(vals, vals, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let mut rgb_vals = vals;
        if _mm_cvtss_f32(alpha_v) != 0.0 {
            rgb_vals = _mm_mul_ps(rgb_vals, _mm_rcp_ps(alpha_v));
        }
        rgb_vals = _mm_min_ps(_mm_max_ps(rgb_vals, zero), one);
        let hi = _mm_shuffle_ps(rgb_vals, alpha_v, mm_shuffle(0, 0, 2, 2));
        let rgb_vals = _mm_shuffle_ps(rgb_vals, hi, mm_shuffle(2, 0, 1, 0));
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(rgb_vals, scale), half));
        let packed = _mm_packs_epi32(idx, idx);
        let packed = _mm_packus_epi16(packed, packed);
        *(out_ptr.add(o_idx) as *mut i32) = _mm_cvtsi128_si32(packed);

        _mm_storeu_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);

        s_idx += 16;
        o_idx += 4;
        i += 1;
    }
}
