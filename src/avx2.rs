use std::arch::x86_64::*;

use crate::srgb;

/// Equivalent to C's mm_shuffle(z, y, x, w).
const fn mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

/// FMA two lane-0 broadcasts into 8 floats of `sums_y_out` (`s_lo`'s 4 taps
/// followed by `s_hi`'s 4 taps), reusing the same 4-tap `coeffs_y` per half.
///
/// Builds the per-tap sample vector by combining `s_lo`/`s_hi` into one ymm
/// first, then a single `vpermilps` to broadcast lane 0 of each 128-bit half.
/// That sequence costs 1 vinsertf128 + 1 vpermilps (2 port-5 uops); the naive
/// "broadcast each half, then insert" sequence costs 2 vpermilps + 1
/// vinsertf128 (3 port-5 uops). Mirrors C's `oil_yacc_fma2_avx2` post-f3d96d0.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yacc_fma2(sums_y_out: *mut f32, s_lo: __m128, s_hi: __m128, cy256: __m256) {
    let s01 = _mm256_insertf128_ps(_mm256_castps128_ps256(s_lo), s_hi, 1);
    let sample = _mm256_permute_ps(s01, 0);
    let sy = _mm256_loadu_ps(sums_y_out);
    let sy = _mm256_fmadd_ps(cy256, sample, sy);
    _mm256_storeu_ps(sums_y_out, sy);
}

/// Single-channel vertical FMA accumulate: load 4 tap floats from
/// `sums_y_out`, FMA with broadcast(`sum`[0]) × `coeffs_y`, store back.
/// Mirrors C's `oil_yacc_fma1_avx2`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yacc_fma1(sums_y_out: *mut f32, sum: __m128, coeffs_y: __m128) {
    let sample = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
    let sy = _mm_loadu_ps(sums_y_out);
    let sy = _mm_fmadd_ps(coeffs_y, sample, sy);
    _mm_storeu_ps(sums_y_out, sy);
}

/// Shift `v` left by one float lane, zero-filling the top lane.
/// Mirrors C's `oil_shift_f_left_avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shift_f_left(v: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 4))
}

/// Consume one output pixel across 4 stride-4 channel ring-buffer slots:
/// gather lane 0 from each of `sums[0..3]`, `sums[4..7]`, `sums[8..11]`,
/// `sums[12..15]` into a packed vector, then shift each slot left to discard
/// the consumed tap. Mirrors C's `oil_consume_ch0_x4_avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn consume_ch0_x4(sums: *mut f32) -> __m128 {
    let f0 = _mm_load_ps(sums);
    let f1 = _mm_load_ps(sums.add(4));
    let f2 = _mm_load_ps(sums.add(8));
    let f3 = _mm_load_ps(sums.add(12));

    let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
    let cd = _mm_shuffle_ps(f2, f3, mm_shuffle(0, 0, 0, 0));
    let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));

    _mm_store_ps(sums, shift_f_left(f0));
    _mm_store_ps(sums.add(4), shift_f_left(f1));
    _mm_store_ps(sums.add(8), shift_f_left(f2));
    _mm_store_ps(sums.add(12), shift_f_left(f3));

    vals
}

/// Clamp `v` to `[0,1]`, multiply by `scale`, round to nearest, truncate to
/// int32. Produces the byte-range index used by sRGB byte packing and LUTs.
/// Mirrors C's `oil_clamp_round_idx_avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn clamp_round_idx(v: __m128, zero: __m128, one: __m128, scale: __m128, half: __m128) -> __m128i {
    let v = _mm_min_ps(_mm_max_ps(v, zero), one);
    _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v, scale), half))
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
        let s_lo = sum;
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        let border1 = *border_ptr.add(i as usize + 1);
        for _j in 0..border1 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
            let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
            sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
            in_idx += 1;
            cx_idx += 4;
        }
        let s_hi = sum;
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        yacc_fma2(sy_ptr.add(sy_idx), s_lo, s_hi, cy256);
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
        let s_lo = sum;
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
        let s_hi = sum;
        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));

        yacc_fma2(sy_ptr.add(sy_idx), s_lo, s_hi, cy256);
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

/// AVX2 downscale for GA (2-byte stride): horizontal x-filtering with
/// alpha-premultiplied gray + 256-bit y-accumulation via `yacc_fma2`.
///
/// Mirrors C's `oil_scale_down_ga_avx2`. The inner pair-tap loop runs only
/// when `border_buf[i] >= 4`; small-border outputs go through a single
/// 128-bit-FMA path. Vertical accumulation packs the two channels into one
/// 256-bit FMA via `yacc_fma2` (sums_y stays channel-major, 8 floats per
/// output pixel).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_ga(
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
    let mut sum_g = _mm_setzero_ps();
    let mut sum_a = _mm_setzero_ps();

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);
        if border >= 4 {
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_a2 = _mm_setzero_ps();
            let mut j = 0;
            while j + 1 < border {
                let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let alpha0 = *i2f.add(*in_ptr.add(in_idx + 1) as usize);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize) * alpha0);
                sum_g = _mm_add_ps(_mm_mul_ps(cx0, s), sum_g);
                let s = _mm_set1_ps(alpha0);
                sum_a = _mm_add_ps(_mm_mul_ps(cx0, s), sum_a);

                let alpha1 = *i2f.add(*in_ptr.add(in_idx + 3) as usize);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize) * alpha1);
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_g2);
                let s = _mm_set1_ps(alpha1);
                sum_a2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_a2);

                in_idx += 4;
                cx_idx += 8;
                j += 2;
            }
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let alpha = *i2f.add(*in_ptr.add(in_idx + 1) as usize);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize) * alpha);
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);
                let s = _mm_set1_ps(alpha);
                sum_a = _mm_add_ps(_mm_mul_ps(cx, s), sum_a);
                in_idx += 2;
                cx_idx += 4;
                j += 1;
            }
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_a = _mm_add_ps(sum_a, sum_a2);
        } else {
            for _ in 0..border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let alpha = *i2f.add(*in_ptr.add(in_idx + 1) as usize);
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize) * alpha);
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);
                let s = _mm_set1_ps(alpha);
                sum_a = _mm_add_ps(_mm_mul_ps(cx, s), sum_a);
                in_idx += 2;
                cx_idx += 4;
            }
        }

        yacc_fma2(sy_ptr.add(sy_idx), sum_g, sum_a, cy256);
        sy_idx += 8;

        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_a = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_a), 4));
    }
}

/// AVX2 downscale for RGB (3-byte stride): horizontal x-filtering with
/// 256-bit-widened pair-tap FMA loop + 128-bit y-accumulation.
///
/// Mirrors C's `oil_scale_down_rgb_avx2` (post-a0c05fc): the pair-tap inner
/// loop packs even/odd taps into the lo/hi lanes of a single 256-bit
/// accumulator per channel, replacing 6 × 128-bit FMAs with 3 × 256-bit FMAs.
/// Carry-over state stays 4-wide; the running 128-bit `sum_*` is sunk into
/// the low lane at pixel start and folded back at pixel end. The trail
/// (border-buf parity tail) and small-border path keep 128-bit FMAs.
///
/// `lut` selects the input gamma path: `s2l_map` for sRGB-linearized
/// `OIL_CS_RGB`, or `i2f_map` for `OIL_CS_RGB_NOGAMMA`. Mirrors the C
/// dispatcher (oil_resample_avx2.c:2042/2067), which calls the same
/// function with different lookup tables.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_rgb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    lut: *const f32,
) {
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());
    let cy256 = _mm256_set_m128(cy, cy);

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
            // Sink running carry-over into lo lane; hi lane starts fresh.
            let zero128 = _mm_setzero_ps();
            let mut sum_r256 =
                _mm256_insertf128_ps(_mm256_castps128_ps256(sum_r), zero128, 1);
            let mut sum_g256 =
                _mm256_insertf128_ps(_mm256_castps128_ps256(sum_g), zero128, 1);
            let mut sum_b256 =
                _mm256_insertf128_ps(_mm256_castps128_ps256(sum_b), zero128, 1);

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm256_loadu_ps(cx_ptr.add(cx_idx));
                let sr = _mm256_set_m128(
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 3) as usize)),
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize)),
                );
                let sg = _mm256_set_m128(
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 4) as usize)),
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize)),
                );
                let sb = _mm256_set_m128(
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 5) as usize)),
                    _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize)),
                );

                sum_r256 = _mm256_fmadd_ps(cx, sr, sum_r256);
                sum_g256 = _mm256_fmadd_ps(cx, sg, sum_g256);
                sum_b256 = _mm256_fmadd_ps(cx, sb, sum_b256);

                in_idx += 6;
                cx_idx += 8;
                j += 2;
            }

            // Fold lo/hi lanes back to 128-bit running sum.
            sum_r = _mm_add_ps(
                _mm256_castps256_ps128(sum_r256),
                _mm256_extractf128_ps(sum_r256, 1),
            );
            sum_g = _mm_add_ps(
                _mm256_castps256_ps128(sum_g256),
                _mm256_extractf128_ps(sum_g256, 1),
            );
            sum_b = _mm_add_ps(
                _mm256_castps256_ps128(sum_b256),
                _mm256_extractf128_ps(sum_b256, 1),
            );

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        }

        // Vertical accumulation: tap-major layout (4 floats per channel).
        // Pack R+G into one 256-bit FMA, leaving B for a 128-bit FMA.
        // Mirrors C's `oil_yacc_fma2_avx2` + `oil_yacc_fma1_avx2` pair.
        yacc_fma2(sy_ptr.add(sy_idx), sum_r, sum_g, cy256);
        yacc_fma1(sy_ptr.add(sy_idx + 8), sum_b, cy);

        sy_idx += 12;

        // Shift the 4-output-pixel pipeline left: lane 0 (just consumed)
        // drops off, lanes 1..3 become 0..2, lane 3 zero-fills.
        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
    }
}

/// AVX2 downscale for RGBX (4-byte stride, no alpha): FMA x-filtering + 256-bit
/// y-accumulation + prefetch.
///
/// Mirrors C's `oil_scale_down_rgbx_avx2` (oil_resample_avx2.c:1586). `lut`
/// selects the input gamma path: `s2l_map` for `OIL_CS_RGBX`, `i2f_map` for
/// `OIL_CS_RGBX_NOGAMMA`. The C dispatcher (line 2065/2074) calls the same
/// function with different lookup tables.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn scale_down_rgbx(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
    lut: *const f32,
) {

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

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_fmadd_ps(cx, s, sum_b);

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_fmadd_ps(cx2, s, sum_r2);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_fmadd_ps(cx2, s, sum_g2);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_fmadd_ps(cx2, s, sum_b2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize));
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

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_fmadd_ps(cx, s, sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_fmadd_ps(cx, s, sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 2) as usize));
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

/// AVX2 downscale shared between RGBA, ARGB, and RGBA_NOGAMMA: horizontal
/// x-filtering with premultiplied alpha + 256-bit FMA y-accumulation.
///
/// Mirrors C's `oil_scale_down_rgba_avx2` (oil_resample_avx2.c:1320).
/// `A_OFF` is the byte offset of the alpha sample within each 4-byte pixel
/// (3 for RGBA, 0 for ARGB); `RGB_OFF` is the offset of the first RGB byte
/// (0 for RGBA, 1 for ARGB). Alpha is always read through `i2f_map`; `lut`
/// is `s2l_map` for gamma callers and `i2f_map` for `RGBA_NOGAMMA`.
/// Mirrors the C dispatcher (line 2056/2062/2071), which calls the same
/// function with different (a_off, rgb_off, lut) triples.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn scale_down_rgba<const A_OFF: usize, const RGB_OFF: usize>(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
    tap: usize,
    lut: *const f32,
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

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + A_OFF) as usize)));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);
                sum_a = _mm_add_ps(cx_a, sum_a);

                let cx2_a = _mm_mul_ps(cx2, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 4 + A_OFF) as usize)));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 4 + RGB_OFF) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_r2);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 4 + RGB_OFF + 1) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_g2);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + 4 + RGB_OFF + 2) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_b2);
                sum_a2 = _mm_add_ps(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + A_OFF) as usize)));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 2) as usize));
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

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + A_OFF) as usize)));

                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);
                let s = _mm_set1_ps(*lut.add(*in_ptr.add(in_idx + RGB_OFF + 2) as usize));
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

/// Write 3 LUT-indexed bytes to `out[0..2]` using the low three int32 lanes
/// of `idx`. Mirrors C's `oil_lut_store3_avx2`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lut_store3_avx2(out: *mut u8, idx: __m128i, lut: *const u8) {
    *out         = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
    *out.add(1)  = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
    *out.add(2)  = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
}

/// AVX2 ring-buffer output shared between gamma RGBA and ARGB. `A_OFF` is the
/// alpha-byte offset, `RGB_OFF` is the first RGB-byte offset.
///
/// Mirrors C's `oil_yscale_out_rgba_avx2`. RGB lanes are clamped, divided by
/// the (clamped) alpha when nonzero, then converted to bytes through `l2s_map`.
/// Alpha is converted directly via `(int)(alpha * 255 + 0.5)`. The consumed
/// tap slot is zeroed, leaving the next tap to slide into place on the next
/// scanline.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn yscale_out_alpha_impl<const A_OFF: usize, const RGB_OFF: usize>(
    sums: &mut [f32], width: u32, out: &mut [u8], tap: usize,
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let z = _mm_setzero_si128();
    let tap_off = tap * 4;

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let vals = _mm_load_ps(s_ptr.add(s_idx + tap_off));

        let alpha_v = _mm_shuffle_ps(vals, vals, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let alpha = _mm_cvtss_f32(alpha_v);

        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = _mm_mul_ps(rgb_vals, _mm_rcp_ps(alpha_v));
        }
        rgb_vals = _mm_min_ps(_mm_max_ps(rgb_vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_mul_ps(rgb_vals, scale));

        let p = out_ptr.add(o_idx);
        lut_store3_avx2(p.add(RGB_OFF), idx, lut);
        *p.add(A_OFF) = (alpha * 255.0 + 0.5) as u8;

        _mm_store_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);

        s_idx += 16;
        o_idx += 4;
    }
}

/// AVX2 ring-buffer output for RGBA (gamma). Mirrors `oil_yscale_out_rgba_avx2`
/// dispatched with `(a_off=3, rgb_off=0)`.
#[target_feature(enable = "avx2")]
pub unsafe fn yscale_out_rgba(sums: &mut [f32], width: u32, out: &mut [u8], tap: usize) {
    yscale_out_alpha_impl::<3, 0>(sums, width, out, tap);
}

/// AVX2 ring-buffer output for ARGB (gamma). Mirrors `oil_yscale_out_rgba_avx2`
/// dispatched with `(a_off=0, rgb_off=1)`.
#[target_feature(enable = "avx2")]
pub unsafe fn yscale_out_argb(sums: &mut [f32], width: u32, out: &mut [u8], tap: usize) {
    yscale_out_alpha_impl::<0, 1>(sums, width, out, tap);
}

/// AVX2 ring-buffer output for RGBX (gamma). Mirrors `oil_yscale_out_rgbx_avx2`.
///
/// One pixel per iteration: clamp the loaded RGB lanes to `[0,1]`, look up the
/// sRGB byte for each through `l2s_map`, force the X byte to 255, and zero the
/// consumed tap slot.
#[target_feature(enable = "avx2")]
pub unsafe fn yscale_out_rgbx(sums: &mut [f32], width: u32, out: &mut [u8], tap: usize) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let z = _mm_setzero_si128();
    let tap_off = tap * 4;

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let vals = _mm_load_ps(s_ptr.add(s_idx + tap_off));
        let vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

        let p = out_ptr.add(o_idx);
        lut_store3_avx2(p, idx, lut);
        *p.add(3) = 255;

        _mm_store_si128(s_ptr.add(s_idx + tap_off) as *mut __m128i, z);

        s_idx += 16;
        o_idx += 4;
    }
}

/// AVX2 ring-buffer output for nonlinear single-channel paths
/// (G, RGB_NOGAMMA, CMYK): clamp each lane-0 sample to `[0,1]`, scale to
/// `[0,255]`, round, and pack to bytes. Mirrors C's
/// `oil_yscale_out_nonlinear_avx2`.
///
/// The ring buffer is laid out channel-major with stride 4: each output
/// channel owns 4 consecutive floats (one per tap phase). We consume lane 0
/// of each slot and shift-left to slide the next tap into place.
#[target_feature(enable = "avx2")]
pub unsafe fn yscale_out_g(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);

    let mut s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 7 < sl_len {
        let vals = consume_ch0_x4(s_ptr);
        let idx = clamp_round_idx(vals, zero, one, scale, half);

        let vals2 = consume_ch0_x4(s_ptr.add(16));
        let idx2 = clamp_round_idx(vals2, zero, one, scale, half);

        let packed = _mm_packs_epi32(idx, idx2);
        let packed = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(out_ptr.add(i) as *mut __m128i, packed);

        s_ptr = s_ptr.add(32);
        i += 8;
    }

    while i + 3 < sl_len {
        let vals = consume_ch0_x4(s_ptr);
        let idx = clamp_round_idx(vals, zero, one, scale, half);

        let packed = _mm_packs_epi32(idx, idx);
        let packed = _mm_packus_epi16(packed, packed);
        *(out_ptr.add(i) as *mut i32) = _mm_cvtsi128_si32(packed);

        s_ptr = s_ptr.add(16);
        i += 4;
    }

    while i < sl_len {
        let v = (*s_ptr).clamp(0.0, 1.0);
        *out_ptr.add(i) = (v * 255.0 + 0.5) as u8;
        let shifted = shift_f_left(_mm_load_ps(s_ptr));
        _mm_store_ps(s_ptr, shifted);
        s_ptr = s_ptr.add(4);
        i += 1;
    }
}
