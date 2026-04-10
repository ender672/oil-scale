use std::arch::aarch64::*;

use crate::srgb;

/// NEON horizontal upscale for RGB.
/// Mirrors oil_xscale_up_rgb: per-channel sliding window with vectorized dot products.
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgb(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;

        smp_r = push_f_neon(smp_r, *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = vgetq_lane_f32::<0>(t2_r);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<0>(t2_g);
            *out_ptr.add(out_idx + 2) = vgetq_lane_f32::<0>(t2_b);
            *out_ptr.add(out_idx + 3) = vgetq_lane_f32::<1>(t2_r);
            *out_ptr.add(out_idx + 4) = vgetq_lane_f32::<1>(t2_g);
            *out_ptr.add(out_idx + 5) = vgetq_lane_f32::<1>(t2_b);

            out_idx += 6;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);

            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for RGB.
/// 4-tap vertical blend, output through l2s LUT.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_rgb(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale_f = (tables.l2s_len - 1) as f32;

    let c0 = vdupq_n_f32(coeffs[0] * scale_f);
    let c1 = vdupq_n_f32(coeffs[1] * scale_f);
    let c2 = vdupq_n_f32(coeffs[2] * scale_f);
    let c3 = vdupq_n_f32(coeffs[3] * scale_f);

    let mut i = 0;
    let out_ptr = out.as_mut_ptr();

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();

    // Process 12 floats at a time (4 RGB pixels)
    while i + 11 < len {
        let sum0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let sum1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 8))),
            c2, vld1q_f32(l2.add(i + 8))),
            c1, vld1q_f32(l1.add(i + 8))),
            c0, vld1q_f32(l0.add(i + 8)));

        let idx0 = vcvtq_s32_f32(sum0);
        let idx1 = vcvtq_s32_f32(sum1);
        let idx2 = vcvtq_s32_f32(sum2);

        *out_ptr.add(i)      = *lut.offset(vgetq_lane_s32::<0>(idx0) as isize);
        *out_ptr.add(i + 1)  = *lut.offset(vgetq_lane_s32::<1>(idx0) as isize);
        *out_ptr.add(i + 2)  = *lut.offset(vgetq_lane_s32::<2>(idx0) as isize);
        *out_ptr.add(i + 3)  = *lut.offset(vgetq_lane_s32::<3>(idx0) as isize);
        *out_ptr.add(i + 4)  = *lut.offset(vgetq_lane_s32::<0>(idx1) as isize);
        *out_ptr.add(i + 5)  = *lut.offset(vgetq_lane_s32::<1>(idx1) as isize);
        *out_ptr.add(i + 6)  = *lut.offset(vgetq_lane_s32::<2>(idx1) as isize);
        *out_ptr.add(i + 7)  = *lut.offset(vgetq_lane_s32::<3>(idx1) as isize);
        *out_ptr.add(i + 8)  = *lut.offset(vgetq_lane_s32::<0>(idx2) as isize);
        *out_ptr.add(i + 9)  = *lut.offset(vgetq_lane_s32::<1>(idx2) as isize);
        *out_ptr.add(i + 10) = *lut.offset(vgetq_lane_s32::<2>(idx2) as isize);
        *out_ptr.add(i + 11) = *lut.offset(vgetq_lane_s32::<3>(idx2) as isize);

        i += 12;
    }

    // Process 4 floats at a time
    while i + 3 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let idx = vcvtq_s32_f32(sum);
        *out_ptr.add(i)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(i + 3) = *lut.offset(vgetq_lane_s32::<3>(idx) as isize);
        i += 4;
    }

    // Scalar tail
    while i < len {
        let val = *coeffs.get_unchecked(0) * scale_f * *l0.add(i)
            + *coeffs.get_unchecked(1) * scale_f * *l1.add(i)
            + *coeffs.get_unchecked(2) * scale_f * *l2.add(i)
            + *coeffs.get_unchecked(3) * scale_f * *l3.add(i);
        *out_ptr.add(i) = *lut.offset(val as isize);
        i += 1;
    }
}

/// NEON downscale for RGB: horizontal x-filtering + y-accumulation.
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 16 {
            // 4-way unroll for large borders (extreme downscale)
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);
            let mut sum_r3 = vdupq_n_f32(0.0);
            let mut sum_g3 = vdupq_n_f32(0.0);
            let mut sum_b3 = vdupq_n_f32(0.0);
            let mut sum_r4 = vdupq_n_f32(0.0);
            let mut sum_g4 = vdupq_n_f32(0.0);
            let mut sum_b4 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 3 < border {
                let cx0 = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx0, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx0, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx0, s);

                let cx1 = vld1q_f32(cx_ptr.add(cx_idx + 4));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx1, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx1, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx1, s);

                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 8));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_r3 = vfmaq_f32(sum_r3, cx2, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 7) as usize));
                sum_g3 = vfmaq_f32(sum_g3, cx2, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 8) as usize));
                sum_b3 = vfmaq_f32(sum_b3, cx2, s);

                let cx3 = vld1q_f32(cx_ptr.add(cx_idx + 12));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 9) as usize));
                sum_r4 = vfmaq_f32(sum_r4, cx3, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 10) as usize));
                sum_g4 = vfmaq_f32(sum_g4, cx3, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 11) as usize));
                sum_b4 = vfmaq_f32(sum_b4, cx3, s);

                in_idx += 12;
                cx_idx += 16;
                j += 4;
            }
            sum_r = vaddq_f32(vaddq_f32(sum_r, sum_r2), vaddq_f32(sum_r3, sum_r4));
            sum_g = vaddq_f32(vaddq_f32(sum_g, sum_g2), vaddq_f32(sum_g3, sum_g4));
            sum_b = vaddq_f32(vaddq_f32(sum_b, sum_b2), vaddq_f32(sum_b3, sum_b4));
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);
                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        } else if border >= 4 {
            // 2-way unroll for moderate borders
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2, s);

                in_idx += 6;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
        } else if border == 1 {
            let cx = vld1q_f32(cx_ptr.add(cx_idx));

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
            sum_r = vfmaq_f32(sum_r, cx, s);

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
            sum_g = vfmaq_f32(sum_g, cx, s);

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
            sum_b = vfmaq_f32(sum_b, cx, s);

            in_idx += 3;
            cx_idx += 4;
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // G channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // B channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        sum_r = vextq_f32::<1>(sum_r, vdupq_n_f32(0.0));
        sum_g = vextq_f32::<1>(sum_g, vdupq_n_f32(0.0));
        sum_b = vextq_f32::<1>(sum_b, vdupq_n_f32(0.0));
    }
}

/// NEON output for downscaled linear RGB: convert sums through l2s LUT.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_rgb(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    let mut s_idx = 0;

    // Process 8 output values at a time
    while i + 7 < sl_len {
        let sp = s_ptr.add(s_idx);

        // First batch of 4
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        let vals = gather_lane0(f0, f1, f2, f3);
        let idx = vcvtq_s32_f32(vmulq_f32(vals, scale));

        // Second batch of 4
        let g0 = vld1q_f32(sp.add(16));
        let g1 = vld1q_f32(sp.add(20));
        let g2 = vld1q_f32(sp.add(24));
        let g3 = vld1q_f32(sp.add(28));

        let vals2 = gather_lane0(g0, g1, g2, g3);
        let idx2 = vcvtq_s32_f32(vmulq_f32(vals2, scale));

        // Interleave LUT lookups from both batches for ILP
        *out_ptr.add(i)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 4) = *lut.offset(vgetq_lane_s32::<0>(idx2) as isize);
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 5) = *lut.offset(vgetq_lane_s32::<1>(idx2) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(i + 6) = *lut.offset(vgetq_lane_s32::<2>(idx2) as isize);
        *out_ptr.add(i + 3) = *lut.offset(vgetq_lane_s32::<3>(idx) as isize);
        *out_ptr.add(i + 7) = *lut.offset(vgetq_lane_s32::<3>(idx2) as isize);

        // Shift all 8 accumulators left
        let zero = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero));
        vst1q_f32(sp.add(16), vextq_f32::<1>(g0, zero));
        vst1q_f32(sp.add(20), vextq_f32::<1>(g1, zero));
        vst1q_f32(sp.add(24), vextq_f32::<1>(g2, zero));
        vst1q_f32(sp.add(28), vextq_f32::<1>(g3, zero));

        s_idx += 32;
        i += 8;
    }

    // Process 4 output values at a time
    while i + 3 < sl_len {
        let sp = s_ptr.add(s_idx);
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        let vals = gather_lane0(f0, f1, f2, f3);
        let idx = vcvtq_s32_f32(vmulq_f32(vals, scale));

        *out_ptr.add(i)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(i + 3) = *lut.offset(vgetq_lane_s32::<3>(idx) as isize);

        let zero = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero));

        s_idx += 16;
        i += 4;
    }

    // Scalar tail
    while i < sl_len {
        let val = *s_ptr.add(s_idx);
        *out_ptr.add(i) = *lut.offset((val * (tables.l2s_len - 1) as f32) as isize);
        // shift_left
        *s_ptr.add(s_idx) = *s_ptr.add(s_idx + 1);
        *s_ptr.add(s_idx + 1) = *s_ptr.add(s_idx + 2);
        *s_ptr.add(s_idx + 2) = *s_ptr.add(s_idx + 3);
        *s_ptr.add(s_idx + 3) = 0.0;
        s_idx += 4;
        i += 1;
    }
}

/// NEON horizontal upscale for RGBA (premultiplied alpha).
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgba(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let mut smp_a = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha_new = *in_ptr.add(in_base + 3) as f32 / 255.0;

        smp_a = push_f_neon(smp_a, alpha_new);
        smp_r = push_f_neon(smp_r, alpha_new * *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, alpha_new * *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, alpha_new * *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            // Transpose [r0,r1] [g0,g1] [b0,b1] [a0,a1] -> [r0,g0,b0,a0] [r1,g1,b1,a1]
            let rg = vzip1q_f32(t2_r, t2_g); // [r0, g0, r1, g1]
            let ba = vzip1q_f32(t2_b, t2_a); // [b0, a0, b1, a1]
            vst1q_f32(out_ptr.add(out_idx), vcombine_f32(vget_low_f32(rg), vget_low_f32(ba)));
            vst1q_f32(out_ptr.add(out_idx + 4), vcombine_f32(vget_high_f32(rg), vget_high_f32(ba)));

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = dot4(smp_a, coeffs);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for RGBA (premultiplied alpha).
/// Processes 4 floats (one RGBA pixel) at a time, un-premultiplies, converts to sRGB.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_rgba(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 3 RGBA pixels (12 floats) at a time
    while i + 11 < len {
        // Vertical blend for pixel 0
        let sum0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        // Vertical blend for pixel 1
        let sum1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        // Vertical blend for pixel 2
        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 8))),
            c2, vld1q_f32(l2.add(i + 8))),
            c1, vld1q_f32(l1.add(i + 8))),
            c0, vld1q_f32(l0.add(i + 8)));

        // Un-premultiply pixel 0
        let a0_v = vdupq_laneq_f32::<3>(sum0);
        let a0_v = vminq_f32(vmaxq_f32(a0_v, zero), one);
        let a0 = vgetq_lane_f32::<0>(a0_v);
        let mut vals0 = sum0;
        if a0 != 0.0 { vals0 = vmulq_f32(vals0, vdivq_f32(vdupq_n_f32(1.0), a0_v)); }
        let clamped0 = vminq_f32(vmaxq_f32(vals0, zero), one);

        // Un-premultiply pixel 1
        let a1_v = vdupq_laneq_f32::<3>(sum1);
        let a1_v = vminq_f32(vmaxq_f32(a1_v, zero), one);
        let a1 = vgetq_lane_f32::<0>(a1_v);
        let mut vals1 = sum1;
        if a1 != 0.0 { vals1 = vmulq_f32(vals1, vdivq_f32(vdupq_n_f32(1.0), a1_v)); }
        let clamped1 = vminq_f32(vmaxq_f32(vals1, zero), one);

        // Un-premultiply pixel 2
        let a2_v = vdupq_laneq_f32::<3>(sum2);
        let a2_v = vminq_f32(vmaxq_f32(a2_v, zero), one);
        let a2 = vgetq_lane_f32::<0>(a2_v);
        let mut vals2 = sum2;
        if a2 != 0.0 { vals2 = vmulq_f32(vals2, vdivq_f32(vdupq_n_f32(1.0), a2_v)); }
        let clamped2 = vminq_f32(vmaxq_f32(vals2, zero), one);

        // Batch convert to l2s indices
        let idx0 = vcvtq_s32_f32(vmulq_f32(clamped0, scale));
        let idx1 = vcvtq_s32_f32(vmulq_f32(clamped1, scale));
        let idx2 = vcvtq_s32_f32(vmulq_f32(clamped2, scale));

        // Batch LUT lookups for RGB + direct alpha writes
        *out_ptr.add(i)      = *lut.offset(vgetq_lane_s32::<0>(idx0) as isize);
        *out_ptr.add(i + 1)  = *lut.offset(vgetq_lane_s32::<1>(idx0) as isize);
        *out_ptr.add(i + 2)  = *lut.offset(vgetq_lane_s32::<2>(idx0) as isize);
        *out_ptr.add(i + 3)  = (a0 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 4)  = *lut.offset(vgetq_lane_s32::<0>(idx1) as isize);
        *out_ptr.add(i + 5)  = *lut.offset(vgetq_lane_s32::<1>(idx1) as isize);
        *out_ptr.add(i + 6)  = *lut.offset(vgetq_lane_s32::<2>(idx1) as isize);
        *out_ptr.add(i + 7)  = (a1 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 8)  = *lut.offset(vgetq_lane_s32::<0>(idx2) as isize);
        *out_ptr.add(i + 9)  = *lut.offset(vgetq_lane_s32::<1>(idx2) as isize);
        *out_ptr.add(i + 10) = *lut.offset(vgetq_lane_s32::<2>(idx2) as isize);
        *out_ptr.add(i + 11) = (a2 * 255.0 + 0.5) as u8;

        i += 12;
    }

    // Process remaining pixels one at a time
    while i < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let alpha_v = vdupq_laneq_f32::<3>(sum);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);

        let mut vals = sum;
        if alpha != 0.0 {
            vals = vmulq_f32(vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        let idx = vcvtq_s32_f32(vmulq_f32(clamped, scale));

        *out_ptr.add(i)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(i + 3) = (alpha * 255.0 + 0.5) as u8;

        i += 4;
    }
}

/// NEON downscale for RGBA: horizontal x-filtering with premultiplied alpha + y-accumulation.
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgba(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);
    let mut sum_a = vdupq_n_f32(0.0);

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
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);
            let mut sum_a2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                let cx2_a = vmulq_f32(cx2, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 7) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2_a, s);

                sum_a2 = vaddq_f32(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
            sum_a = vaddq_f32(sum_a, sum_a2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // G channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // B channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // A channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_a);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // shift_left for each channel
        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
        sum_a = vextq_f32::<1>(sum_a, zero);
    }
}

/// NEON output for downscaled RGBA: un-premultiply, convert RGB through l2s LUT, alpha to byte.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_rgba(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx);

        // Load 4 accumulators for this pixel: [R0..R3], [G0..G3], [B0..B3], [A0..A3]
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        // Gather first element of each accumulator: {R, G, B, A}
        let vals = gather_lane0(f0, f1, f2, f3);

        // Clamp alpha to [0, 1]
        let alpha_v = vdupq_laneq_f32::<3>(vals);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);

        // Divide RGB by alpha (skip if alpha == 0)
        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = vmulq_f32(rgb_vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }

        // Clamp RGB to [0, 1] and compute l2s_map indices
        rgb_vals = vminq_f32(vmaxq_f32(rgb_vals, zero), one);
        let idx = vcvtq_s32_f32(vmulq_f32(rgb_vals, scale));

        *out_ptr.add(o_idx)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(o_idx + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(o_idx + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(o_idx + 3) = (alpha * 255.0 + 0.5) as u8;

        // Shift all 4 accumulators left
        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero_v));

        s_idx += 16;
        o_idx += 4;
    }
}

// --- ARGB NEON ---

/// NEON horizontal upscale for ARGB (alpha-first byte order).
/// Alpha at input byte 0, RGB at input bytes 1-3. Internal float layout is [R,G,B,A].
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_argb(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let mut smp_a = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha_new = *in_ptr.add(in_base) as f32 / 255.0;

        smp_a = push_f_neon(smp_a, alpha_new);
        smp_r = push_f_neon(smp_r, alpha_new * *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_g = push_f_neon(smp_g, alpha_new * *s2l.add(*in_ptr.add(in_base + 2) as usize));
        smp_b = push_f_neon(smp_b, alpha_new * *s2l.add(*in_ptr.add(in_base + 3) as usize));

        let mut j = *border_ptr.add(i);

        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            let rg = vzip1q_f32(t2_r, t2_g);
            let ba = vzip1q_f32(t2_b, t2_a);
            vst1q_f32(out_ptr.add(out_idx), vcombine_f32(vget_low_f32(rg), vget_low_f32(ba)));
            vst1q_f32(out_ptr.add(out_idx + 4), vcombine_f32(vget_high_f32(rg), vget_high_f32(ba)));

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = dot4(smp_a, coeffs);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for ARGB (premultiplied alpha).
/// Same blend as RGBA but writes output bytes as [A,R,G,B].
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_argb(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    while i + 11 < len {
        let sum0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let sum1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 8))),
            c2, vld1q_f32(l2.add(i + 8))),
            c1, vld1q_f32(l1.add(i + 8))),
            c0, vld1q_f32(l0.add(i + 8)));

        let a0_v = vdupq_laneq_f32::<3>(sum0);
        let a0_v = vminq_f32(vmaxq_f32(a0_v, zero), one);
        let a0 = vgetq_lane_f32::<0>(a0_v);
        let mut vals0 = sum0;
        if a0 != 0.0 { vals0 = vmulq_f32(vals0, vdivq_f32(vdupq_n_f32(1.0), a0_v)); }
        let clamped0 = vminq_f32(vmaxq_f32(vals0, zero), one);

        let a1_v = vdupq_laneq_f32::<3>(sum1);
        let a1_v = vminq_f32(vmaxq_f32(a1_v, zero), one);
        let a1 = vgetq_lane_f32::<0>(a1_v);
        let mut vals1 = sum1;
        if a1 != 0.0 { vals1 = vmulq_f32(vals1, vdivq_f32(vdupq_n_f32(1.0), a1_v)); }
        let clamped1 = vminq_f32(vmaxq_f32(vals1, zero), one);

        let a2_v = vdupq_laneq_f32::<3>(sum2);
        let a2_v = vminq_f32(vmaxq_f32(a2_v, zero), one);
        let a2 = vgetq_lane_f32::<0>(a2_v);
        let mut vals2 = sum2;
        if a2 != 0.0 { vals2 = vmulq_f32(vals2, vdivq_f32(vdupq_n_f32(1.0), a2_v)); }
        let clamped2 = vminq_f32(vmaxq_f32(vals2, zero), one);

        let idx0 = vcvtq_s32_f32(vmulq_f32(clamped0, scale));
        let idx1 = vcvtq_s32_f32(vmulq_f32(clamped1, scale));
        let idx2 = vcvtq_s32_f32(vmulq_f32(clamped2, scale));

        // ARGB output: [A, R, G, B]
        *out_ptr.add(i)      = (a0 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 1)  = *lut.offset(vgetq_lane_s32::<0>(idx0) as isize);
        *out_ptr.add(i + 2)  = *lut.offset(vgetq_lane_s32::<1>(idx0) as isize);
        *out_ptr.add(i + 3)  = *lut.offset(vgetq_lane_s32::<2>(idx0) as isize);
        *out_ptr.add(i + 4)  = (a1 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 5)  = *lut.offset(vgetq_lane_s32::<0>(idx1) as isize);
        *out_ptr.add(i + 6)  = *lut.offset(vgetq_lane_s32::<1>(idx1) as isize);
        *out_ptr.add(i + 7)  = *lut.offset(vgetq_lane_s32::<2>(idx1) as isize);
        *out_ptr.add(i + 8)  = (a2 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 9)  = *lut.offset(vgetq_lane_s32::<0>(idx2) as isize);
        *out_ptr.add(i + 10) = *lut.offset(vgetq_lane_s32::<1>(idx2) as isize);
        *out_ptr.add(i + 11) = *lut.offset(vgetq_lane_s32::<2>(idx2) as isize);

        i += 12;
    }

    while i < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let alpha_v = vdupq_laneq_f32::<3>(sum);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);

        let mut vals = sum;
        if alpha != 0.0 {
            vals = vmulq_f32(vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        let idx = vcvtq_s32_f32(vmulq_f32(clamped, scale));

        // ARGB output: [A, R, G, B]
        *out_ptr.add(i)     = (alpha * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 3) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);

        i += 4;
    }
}

/// NEON downscale for ARGB: horizontal x-filtering with premultiplied alpha + y-accumulation.
/// Alpha at input byte 0, RGB at input bytes 1-3.
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_argb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);
    let mut sum_a = vdupq_n_f32(0.0);

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
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);
            let mut sum_a2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                // ARGB: alpha at byte 0
                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                // Second pixel: alpha at byte 4
                let cx2_a = vmulq_f32(cx2, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 7) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2_a, s);

                sum_a2 = vaddq_f32(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
            sum_a = vaddq_f32(sum_a, sum_a2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize)));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);

                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // G channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // B channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // A channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_a);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        // shift_left for each channel
        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
        sum_a = vextq_f32::<1>(sum_a, zero);
    }
}

/// NEON output for downscaled ARGB: un-premultiply, convert RGB through l2s LUT, alpha to byte.
/// Writes output bytes as [A,R,G,B].
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_argb(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx);

        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        let vals = gather_lane0(f0, f1, f2, f3);

        let alpha_v = vdupq_laneq_f32::<3>(vals);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);

        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = vmulq_f32(rgb_vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }

        rgb_vals = vminq_f32(vmaxq_f32(rgb_vals, zero), one);
        let idx = vcvtq_s32_f32(vmulq_f32(rgb_vals, scale));

        // ARGB output: [A, R, G, B]
        *out_ptr.add(o_idx)     = (alpha * 255.0 + 0.5) as u8;
        *out_ptr.add(o_idx + 1) = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(o_idx + 2) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(o_idx + 3) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);

        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero_v));

        s_idx += 16;
        o_idx += 4;
    }
}

// --- RGBX NEON ---

/// NEON horizontal upscale for RGBX.
/// Like RGBA but 4th component is always 1.0 and RGB is not premultiplied by alpha.
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgbx(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;

        smp_r = push_f_neon(smp_r, *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = vgetq_lane_f32::<0>(t2_r);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<0>(t2_g);
            *out_ptr.add(out_idx + 2) = vgetq_lane_f32::<0>(t2_b);
            *out_ptr.add(out_idx + 3) = 1.0;
            *out_ptr.add(out_idx + 4) = vgetq_lane_f32::<1>(t2_r);
            *out_ptr.add(out_idx + 5) = vgetq_lane_f32::<1>(t2_g);
            *out_ptr.add(out_idx + 6) = vgetq_lane_f32::<1>(t2_b);
            *out_ptr.add(out_idx + 7) = 1.0;

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = 1.0;

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for RGBX.
/// No alpha un-premultiply; RGB through l2s LUT, X byte always 255.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_rgbx(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale_f = (tables.l2s_len - 1) as f32;

    let c0 = vdupq_n_f32(coeffs[0] * scale_f);
    let c1 = vdupq_n_f32(coeffs[1] * scale_f);
    let c2 = vdupq_n_f32(coeffs[2] * scale_f);
    let c3 = vdupq_n_f32(coeffs[3] * scale_f);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 12 floats at a time (3 RGBX pixels)
    while i + 11 < len {
        let sum0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let sum1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 8))),
            c2, vld1q_f32(l2.add(i + 8))),
            c1, vld1q_f32(l1.add(i + 8))),
            c0, vld1q_f32(l0.add(i + 8)));

        let idx0 = vcvtq_s32_f32(sum0);
        let idx1 = vcvtq_s32_f32(sum1);
        let idx2 = vcvtq_s32_f32(sum2);

        *out_ptr.add(i)      = *lut.offset(vgetq_lane_s32::<0>(idx0) as isize);
        *out_ptr.add(i + 1)  = *lut.offset(vgetq_lane_s32::<1>(idx0) as isize);
        *out_ptr.add(i + 2)  = *lut.offset(vgetq_lane_s32::<2>(idx0) as isize);
        *out_ptr.add(i + 3)  = 255;
        *out_ptr.add(i + 4)  = *lut.offset(vgetq_lane_s32::<0>(idx1) as isize);
        *out_ptr.add(i + 5)  = *lut.offset(vgetq_lane_s32::<1>(idx1) as isize);
        *out_ptr.add(i + 6)  = *lut.offset(vgetq_lane_s32::<2>(idx1) as isize);
        *out_ptr.add(i + 7)  = 255;
        *out_ptr.add(i + 8)  = *lut.offset(vgetq_lane_s32::<0>(idx2) as isize);
        *out_ptr.add(i + 9)  = *lut.offset(vgetq_lane_s32::<1>(idx2) as isize);
        *out_ptr.add(i + 10) = *lut.offset(vgetq_lane_s32::<2>(idx2) as isize);
        *out_ptr.add(i + 11) = 255;

        i += 12;
    }

    // Process 4 floats at a time (1 RGBX pixel)
    while i + 3 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let idx = vcvtq_s32_f32(sum);
        *out_ptr.add(i)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(i + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(i + 3) = 255;
        i += 4;
    }
}

/// NEON downscale for RGBX: horizontal x-filtering with X=1.0 + y-accumulation.
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgbx(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 16 {
            // 4-way unroll for large borders (extreme downscale)
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);
            let mut sum_r3 = vdupq_n_f32(0.0);
            let mut sum_g3 = vdupq_n_f32(0.0);
            let mut sum_b3 = vdupq_n_f32(0.0);
            let mut sum_r4 = vdupq_n_f32(0.0);
            let mut sum_g4 = vdupq_n_f32(0.0);
            let mut sum_b4 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 3 < border {
                let cx0 = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx0, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx0, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx0, s);

                let cx1 = vld1q_f32(cx_ptr.add(cx_idx + 4));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx1, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx1, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx1, s);

                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 8));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 8) as usize));
                sum_r3 = vfmaq_f32(sum_r3, cx2, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 9) as usize));
                sum_g3 = vfmaq_f32(sum_g3, cx2, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 10) as usize));
                sum_b3 = vfmaq_f32(sum_b3, cx2, s);

                let cx3 = vld1q_f32(cx_ptr.add(cx_idx + 12));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 12) as usize));
                sum_r4 = vfmaq_f32(sum_r4, cx3, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 13) as usize));
                sum_g4 = vfmaq_f32(sum_g4, cx3, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 14) as usize));
                sum_b4 = vfmaq_f32(sum_b4, cx3, s);

                in_idx += 16;
                cx_idx += 16;
                j += 4;
            }
            sum_r = vaddq_f32(vaddq_f32(sum_r, sum_r2), vaddq_f32(sum_r3, sum_r4));
            sum_g = vaddq_f32(vaddq_f32(sum_g, sum_g2), vaddq_f32(sum_g3, sum_g4));
            sum_b = vaddq_f32(vaddq_f32(sum_b, sum_b2), vaddq_f32(sum_b3, sum_b4));
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);
                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        } else if border >= 4 {
            // 2-way unroll for moderate borders
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2, s);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
        } else if border == 1 {
            let cx = vld1q_f32(cx_ptr.add(cx_idx));

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
            sum_r = vfmaq_f32(sum_r, cx, s);

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
            sum_g = vfmaq_f32(sum_g, cx, s);

            let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
            sum_b = vfmaq_f32(sum_b, cx, s);

            in_idx += 4;
            cx_idx += 4;
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);

                let s = vdupq_n_f32(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // G channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // B channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Skip X channel y-accumulation (X is always 1.0, output always 255)
        sy_idx += 4;

        // shift_left for each channel
        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
    }
}

/// NEON output for downscaled RGBX: convert RGB through l2s LUT, X byte always 255.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_rgbx(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = vdupq_n_f32((tables.l2s_len - 1) as f32);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx);

        // Load 3 accumulators for this pixel: [R0..R3], [G0..G3], [B0..B3]
        // X accumulator is unused (output always 255)
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));

        // Gather first element of each accumulator: {R, G, B, _}
        let vals = gather_lane0(f0, f1, f2, f2);

        // Clamp RGB to [0, 1] and compute l2s_map indices
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        let idx = vcvtq_s32_f32(vmulq_f32(clamped, scale));

        *out_ptr.add(o_idx)     = *lut.offset(vgetq_lane_s32::<0>(idx) as isize);
        *out_ptr.add(o_idx + 1) = *lut.offset(vgetq_lane_s32::<1>(idx) as isize);
        *out_ptr.add(o_idx + 2) = *lut.offset(vgetq_lane_s32::<2>(idx) as isize);
        *out_ptr.add(o_idx + 3) = 255;

        // Shift R, G, B accumulators left (skip X)
        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));

        s_idx += 16;
        o_idx += 4;
    }
}

// --- Grayscale (G) NEON ---

/// NEON horizontal upscale for G (grayscale).
/// Single sliding window with vectorized dot products.
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_g(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let mut smp = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        smp = push_f_neon(smp, *i2f.add(*in_ptr.add(i) as usize));

        let mut j = *border_ptr.add(i);

        // Process quads of outputs: transpose 4 coefficient vectors, broadcast
        // each sample element, FMA to get 4 dot products, vector store.
        while j >= 4 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));
            let c2 = vld1q_f32(coeff_ptr.add(coeff_idx + 8));
            let c3 = vld1q_f32(coeff_ptr.add(coeff_idx + 12));

            // Transpose 4x4: [c0,c1,c2,c3] rows -> columns
            let t01_lo = vzip1q_f32(c0, c1); // [c0_0, c1_0, c0_1, c1_1]
            let t01_hi = vzip2q_f32(c0, c1); // [c0_2, c1_2, c0_3, c1_3]
            let t23_lo = vzip1q_f32(c2, c3); // [c2_0, c3_0, c2_1, c3_1]
            let t23_hi = vzip2q_f32(c2, c3); // [c2_2, c3_2, c2_3, c3_3]
            let row0 = vcombine_f32(vget_low_f32(t01_lo), vget_low_f32(t23_lo));
            let row1 = vcombine_f32(vget_high_f32(t01_lo), vget_high_f32(t23_lo));
            let row2 = vcombine_f32(vget_low_f32(t01_hi), vget_low_f32(t23_hi));
            let row3 = vcombine_f32(vget_high_f32(t01_hi), vget_high_f32(t23_hi));

            // Broadcast each sample element
            let s0 = vdupq_laneq_f32::<0>(smp);
            let s1 = vdupq_laneq_f32::<1>(smp);
            let s2 = vdupq_laneq_f32::<2>(smp);
            let s3 = vdupq_laneq_f32::<3>(smp);

            // result = s0*row0 + s1*row1 + s2*row2 + s3*row3
            let result = vfmaq_f32(vfmaq_f32(vfmaq_f32(
                vmulq_f32(s3, row3),
                s2, row2),
                s1, row1),
                s0, row0);
            vst1q_f32(out_ptr.add(out_idx), result);

            out_idx += 4;
            coeff_idx += 16;
            j -= 4;
        }

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));
            let t2 = dot4x2(smp, c0, c1);
            *out_ptr.add(out_idx) = vgetq_lane_f32::<0>(t2);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<1>(t2);
            out_idx += 2;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));
            *out_ptr.add(out_idx) = dot4(smp, coeffs);
            out_idx += 1;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for G (grayscale).
/// 4-tap vertical blend with FMA + NEON packing.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_g(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;

    // Process 16 pixels at a time
    while i + 15 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));

        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        let idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum2, scale), half));

        let sum3 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 8))),
            c2, vld1q_f32(l2.add(i + 8))),
            c1, vld1q_f32(l1.add(i + 8))),
            c0, vld1q_f32(l0.add(i + 8)));
        let idx3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum3, scale), half));

        let sum4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 12))),
            c2, vld1q_f32(l2.add(i + 12))),
            c1, vld1q_f32(l1.add(i + 12))),
            c0, vld1q_f32(l0.add(i + 12)));
        let idx4 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum4, scale), half));

        let packed12 = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
        let packed34 = vcombine_s16(vqmovn_s32(idx3), vqmovn_s32(idx4));
        let result = vcombine_u8(vqmovun_s16(packed12), vqmovun_s16(packed34));
        vst1q_u8(out_ptr.add(i), result);
        i += 16;
    }

    // Process 8 pixels at a time
    while i + 7 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));

        let sum2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));
        let idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum2, scale), half));

        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
        let result = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        vst1_u8(out_ptr.add(i), vget_low_u8(result));
        i += 8;
    }

    // Process 4 pixels at a time
    while i + 3 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx));
        let result = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        *(out_ptr.add(i) as *mut u32) = vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(result)));
        i += 4;
    }

    // Scalar tail
    while i < len {
        let s = coeffs[0] * *l0.add(i) + coeffs[1] * *l1.add(i)
            + coeffs[2] * *l2.add(i) + coeffs[3] * *l3.add(i);
        let s = s.clamp(0.0, 1.0);
        *out_ptr.add(i) = (s * 255.0 + 0.5) as u8;
        i += 1;
    }
}

/// NEON downscale for G: horizontal x-filtering + y-accumulation.
#[target_feature(enable = "neon")]
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
    let cy = vld1q_f32(coeffs_y.as_ptr());
    let mut sum = vdupq_n_f32(0.0);
    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 8 {
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);
            let mut sum4 = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 3 < border {
                let s0 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                let s1 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let s2 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                let s3 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                let cx0 = vld1q_f32(cx_ptr.add(cx_idx));
                let cx1 = vld1q_f32(cx_ptr.add(cx_idx + 4));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 8));
                let cx3 = vld1q_f32(cx_ptr.add(cx_idx + 12));
                sum = vfmaq_f32(sum, cx0, s0);
                sum2 = vfmaq_f32(sum2, cx1, s1);
                sum3 = vfmaq_f32(sum3, cx2, s2);
                sum4 = vfmaq_f32(sum4, cx3, s3);
                in_idx += 4;
                cx_idx += 16;
                j += 4;
            }
            sum = vaddq_f32(vaddq_f32(sum, sum2), vaddq_f32(sum3, sum4));
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum = vfmaq_f32(sum, cx, s);
                in_idx += 1;
                cx_idx += 4;
                j += 1;
            }
        } else {
            for _ in 0..border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum = vfmaq_f32(sum, cx, s);
                in_idx += 1;
                cx_idx += 4;
            }
        }

        let sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample_y = vdupq_laneq_f32::<0>(sum);
        let sy_new = vfmaq_f32(sy, cy, sample_y);
        vst1q_f32(sy_ptr.add(sy_idx), sy_new);
        sy_idx += 4;

        sum = vextq_f32::<1>(sum, vdupq_n_f32(0.0));
    }
}

/// NEON vertical output for G downscale.
/// Extracts first element from each 4-element accumulator, clamps, converts to byte.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_g(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    let mut s_idx = 0;

    // Process 4 output values at a time
    while i + 3 < sl_len {
        let sp = s_ptr.add(s_idx);
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        // Extract first float from each 4-element accumulator
        let vals = gather_lane0(f0, f1, f2, f3);

        // Clamp to [0, 1], scale to [0, 255], add 0.5, truncate to int, pack to bytes
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale), half));
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx));
        let result = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        *(out_ptr.add(i) as *mut u32) = vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(result)));

        // Shift all 4 accumulators left
        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero_v));

        s_idx += 16;
        i += 4;
    }

    // Scalar tail
    while i < sl_len {
        let val = *s_ptr.add(s_idx);
        let val = val.clamp(0.0, 1.0);
        *out_ptr.add(i) = (val * 255.0 + 0.5) as u8;
        // shift_left
        *s_ptr.add(s_idx) = *s_ptr.add(s_idx + 1);
        *s_ptr.add(s_idx + 1) = *s_ptr.add(s_idx + 2);
        *s_ptr.add(s_idx + 2) = *s_ptr.add(s_idx + 3);
        *s_ptr.add(s_idx + 3) = 0.0;
        s_idx += 4;
        i += 1;
    }
}

// --- CMYK NEON ---

/// NEON horizontal upscale for CMYK.
/// Interleaved layout: each smpN = [C, M, Y, K] for one tap position.
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_cmyk(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let inv255 = vdupq_n_f32(1.0 / 255.0);
    let mut smp0;
    let mut smp1 = vdupq_n_f32(0.0);
    let mut smp2 = vdupq_n_f32(0.0);
    let mut smp3 = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        // Load 4 bytes [C,M,Y,K], unpack to 4 floats, divide by 255
        let raw = *(in_ptr.add(i * 4) as *const u32);
        let b = vreinterpret_u8_u32(vdup_n_u32(raw));
        let h = vmovl_u8(b); // u8 -> u16
        let w = vmovl_u16(vget_low_u16(h)); // u16 -> u32
        let f = vcvtq_f32_u32(w);

        smp0 = smp1;
        smp1 = smp2;
        smp2 = smp3;
        smp3 = vmulq_f32(f, inv255);

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let result0 = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(smp0, vdupq_laneq_f32::<0>(c0)),
                    vmulq_f32(smp1, vdupq_laneq_f32::<1>(c0)),
                ),
                vaddq_f32(
                    vmulq_f32(smp2, vdupq_laneq_f32::<2>(c0)),
                    vmulq_f32(smp3, vdupq_laneq_f32::<3>(c0)),
                ),
            );

            let result1 = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(smp0, vdupq_laneq_f32::<0>(c1)),
                    vmulq_f32(smp1, vdupq_laneq_f32::<1>(c1)),
                ),
                vaddq_f32(
                    vmulq_f32(smp2, vdupq_laneq_f32::<2>(c1)),
                    vmulq_f32(smp3, vdupq_laneq_f32::<3>(c1)),
                ),
            );

            vst1q_f32(out_ptr.add(out_idx), result0);
            vst1q_f32(out_ptr.add(out_idx + 4), result1);

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let c = vld1q_f32(coeff_ptr.add(coeff_idx));

            let result = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(smp0, vdupq_laneq_f32::<0>(c)),
                    vmulq_f32(smp1, vdupq_laneq_f32::<1>(c)),
                ),
                vaddq_f32(
                    vmulq_f32(smp2, vdupq_laneq_f32::<2>(c)),
                    vmulq_f32(smp3, vdupq_laneq_f32::<3>(c)),
                ),
            );

            vst1q_f32(out_ptr.add(out_idx), result);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON downscale for CMYK: horizontal x-filtering with i2f + y-accumulation.
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_cmyk(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_c = vdupq_n_f32(0.0);
    let mut sum_m = vdupq_n_f32(0.0);
    let mut sum_yc = vdupq_n_f32(0.0);
    let mut sum_k = vdupq_n_f32(0.0);

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
            let mut sum_c2 = vdupq_n_f32(0.0);
            let mut sum_m2 = vdupq_n_f32(0.0);
            let mut sum_yc2 = vdupq_n_f32(0.0);
            let mut sum_k2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = vfmaq_f32(sum_c, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = vfmaq_f32(sum_m, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = vfmaq_f32(sum_yc, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = vfmaq_f32(sum_k, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_c2 = vfmaq_f32(sum_c2, cx2, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_m2 = vfmaq_f32(sum_m2, cx2, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_yc2 = vfmaq_f32(sum_yc2, cx2, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 7) as usize));
                sum_k2 = vfmaq_f32(sum_k2, cx2, s);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = vfmaq_f32(sum_c, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = vfmaq_f32(sum_m, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = vfmaq_f32(sum_yc, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = vfmaq_f32(sum_k, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_c = vaddq_f32(sum_c, sum_c2);
            sum_m = vaddq_f32(sum_m, sum_m2);
            sum_yc = vaddq_f32(sum_yc, sum_yc2);
            sum_k = vaddq_f32(sum_k, sum_k2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = vfmaq_f32(sum_c, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = vfmaq_f32(sum_m, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = vfmaq_f32(sum_yc, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = vfmaq_f32(sum_k, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // C channel
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_c);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // M channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_m);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Y channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_yc);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // K channel
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_k);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        let zero = vdupq_n_f32(0.0);
        sum_c = vextq_f32::<1>(sum_c, zero);
        sum_m = vextq_f32::<1>(sum_m, zero);
        sum_yc = vextq_f32::<1>(sum_yc, zero);
        sum_k = vextq_f32::<1>(sum_k, zero);
    }
}

// --- Grayscale + Alpha (GA) NEON ---

/// NEON horizontal upscale for GA (grayscale with premultiplied alpha).
/// Two sliding windows: gray (premultiplied) and alpha.
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_ga(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_a = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 2;
        let alpha = *in_ptr.add(in_base + 1) as f32 / 255.0;
        smp_a = push_f_neon(smp_a, alpha);
        let gray = *in_ptr.add(in_base) as f32 / 255.0;
        smp_g = push_f_neon(smp_g, alpha * gray);

        let mut j = *border_ptr.add(i);

        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            *out_ptr.add(out_idx)     = vgetq_lane_f32::<0>(t2_g);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<0>(t2_a);
            *out_ptr.add(out_idx + 2) = vgetq_lane_f32::<1>(t2_g);
            *out_ptr.add(out_idx + 3) = vgetq_lane_f32::<1>(t2_a);

            out_idx += 4;
            coeff_idx += 8;
            j -= 2;
        }

        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_a, coeffs);

            out_idx += 2;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for GA.
/// Vectorized un-premultiply, clamp, and pack to u8.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_ga(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let v255 = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    // Mask selecting alpha channel positions (1 and 3) in [g, a, g, a]
    let mask_arr: [u32; 4] = [0, 0xFFFFFFFF, 0, 0xFFFFFFFF];
    let alpha_mask = vld1q_u32(mask_arr.as_ptr());

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 4 pixels (8 floats = 2 x float32x4_t) at a time
    while i + 7 < len {
        // Vertical blend for first 2 pixels
        let sum_lo = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));
        // Vertical blend for next 2 pixels
        let sum_hi = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));

        // Un-premultiply and pack first pair
        let result_lo = yscale_up_ga_unpremultiply(sum_lo, alpha_mask, zero, one, v255, half);
        // Un-premultiply and pack second pair
        let result_hi = yscale_up_ga_unpremultiply(sum_hi, alpha_mask, zero, one, v255, half);

        // Pack i32 -> i16 -> u8
        let packed16 = vcombine_s16(vqmovn_s32(result_lo), vqmovn_s32(result_hi));
        let packed8 = vcombine_u8(vqmovun_s16(packed16), vqmovun_s16(packed16));

        // Store 8 bytes (4 pixels x 2 channels)
        std::ptr::write_unaligned(
            out_ptr.add(i) as *mut u64,
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(packed8)),
        );

        i += 8;
    }

    // Process 2 pixels (4 floats) at a time
    while i + 3 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let result = yscale_up_ga_unpremultiply(sum, alpha_mask, zero, one, v255, half);

        // Pack and store 4 bytes (2 pixels)
        let packed16 = vcombine_s16(vqmovn_s32(result), vqmovn_s32(result));
        let packed8 = vcombine_u8(vqmovun_s16(packed16), vqmovun_s16(packed16));
        std::ptr::write_unaligned(
            out_ptr.add(i) as *mut u32,
            vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(packed8))),
        );

        i += 4;
    }

    // Process remaining single pixel (2 floats)
    while i + 1 < len {
        let g = coeffs[0] * *l0.add(i) + coeffs[1] * *l1.add(i)
            + coeffs[2] * *l2.add(i) + coeffs[3] * *l3.add(i);
        let a = coeffs[0] * *l0.add(i + 1) + coeffs[1] * *l1.add(i + 1)
            + coeffs[2] * *l2.add(i + 1) + coeffs[3] * *l3.add(i + 1);
        let alpha = a.clamp(0.0, 1.0);
        let mut gray = g;
        if alpha != 0.0 {
            gray /= alpha;
        }
        let gray = gray.clamp(0.0, 1.0);
        *out_ptr.add(i) = (gray * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 1) = (alpha * 255.0 + 0.5) as u8;
        i += 2;
    }
}

/// Vectorized un-premultiply, clamp, scale, and convert to i32 for a [g, a, g, a] vector.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn yscale_up_ga_unpremultiply(
    sum: float32x4_t,
    alpha_mask: uint32x4_t,
    zero: float32x4_t,
    one: float32x4_t,
    v255: float32x4_t,
    half: float32x4_t,
) -> int32x4_t {
    // Spread alpha to both channels: [g0, a0, g1, a1] -> [a0, a0, a1, a1]
    let alpha_spread = vtrn2q_f32(sum, sum);
    let alpha_clamped = vminq_f32(vmaxq_f32(alpha_spread, zero), one);

    // Safe division: where alpha == 0, substitute 1.0 to avoid inf/nan
    let nonzero = vmvnq_u32(vceqq_f32(alpha_clamped, zero));
    let safe_alpha = vbslq_f32(nonzero, alpha_clamped, one);
    let divided = vdivq_f32(sum, safe_alpha);
    let clamped = vminq_f32(vmaxq_f32(divided, zero), one);

    // Merge: gray channels from divided, alpha channels from original clamped
    let result = vbslq_f32(alpha_mask, alpha_clamped, clamped);

    // Scale to [0, 255] with rounding and convert to i32
    vcvtq_s32_f32(vaddq_f32(vmulq_f32(result, v255), half))
}

/// NEON downscale for GA: horizontal x-filtering with premultiplied alpha + y-accumulation.
#[target_feature(enable = "neon")]
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
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_a = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 8 {
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_g3 = vdupq_n_f32(0.0);
            let mut sum_g4 = vdupq_n_f32(0.0);
            let mut sum_a2 = vdupq_n_f32(0.0);
            let mut sum_a3 = vdupq_n_f32(0.0);
            let mut sum_a4 = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 3 < border {
                let cx0 = vld1q_f32(cx_ptr.add(cx_idx));
                let cx1 = vld1q_f32(cx_ptr.add(cx_idx + 4));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 8));
                let cx3 = vld1q_f32(cx_ptr.add(cx_idx + 12));

                let a0 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let a1 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                let a2 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                let a3 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 7) as usize));

                let cx_a0 = vmulq_f32(cx0, a0);
                let cx_a1 = vmulq_f32(cx1, a1);
                let cx_a2 = vmulq_f32(cx2, a2);
                let cx_a3 = vmulq_f32(cx3, a3);

                let g0 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                let g1 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                let g2 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                let g3 = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 6) as usize));

                sum_g  = vfmaq_f32(sum_g, cx_a0, g0);
                sum_g2 = vfmaq_f32(sum_g2, cx_a1, g1);
                sum_g3 = vfmaq_f32(sum_g3, cx_a2, g2);
                sum_g4 = vfmaq_f32(sum_g4, cx_a3, g3);

                sum_a  = vaddq_f32(cx_a0, sum_a);
                sum_a2 = vaddq_f32(cx_a1, sum_a2);
                sum_a3 = vaddq_f32(cx_a2, sum_a3);
                sum_a4 = vaddq_f32(cx_a3, sum_a4);

                in_idx += 8;
                cx_idx += 16;
                j += 4;
            }
            sum_g = vaddq_f32(vaddq_f32(sum_g, sum_g2), vaddq_f32(sum_g3, sum_g4));
            sum_a = vaddq_f32(vaddq_f32(sum_a, sum_a2), vaddq_f32(sum_a3, sum_a4));
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let a = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let cx_a = vmulq_f32(cx, a);
                let g = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, g);
                sum_a = vaddq_f32(cx_a, sum_a);
                in_idx += 2;
                cx_idx += 4;
                j += 1;
            }
        } else {
            for _ in 0..border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let a = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let cx_a = vmulq_f32(cx, a);
                let g = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, g);
                sum_a = vaddq_f32(cx_a, sum_a);
                in_idx += 2;
                cx_idx += 4;
            }
        }

        // Accumulate gray into y sums
        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Accumulate alpha into y sums
        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_a);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        let zero = vdupq_n_f32(0.0);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_a = vextq_f32::<1>(sum_a, zero);
    }
}

/// NEON output for downscaled GA: un-premultiply gray, convert to bytes.
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_ga(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx);

        // Load 2 accumulators: [G0..G3], [A0..A3]
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));

        // Extract first element of each: gray, alpha
        let gray_val = vgetq_lane_f32::<0>(f0);
        let alpha_val = vgetq_lane_f32::<0>(f1);

        // Clamp alpha
        let alpha = alpha_val.clamp(0.0, 1.0);

        // Un-premultiply gray
        let mut gray = gray_val;
        if alpha != 0.0 {
            gray /= alpha;
        }
        let gray = gray.clamp(0.0, 1.0);

        *out_ptr.add(o_idx) = (gray * 255.0 + 0.5) as u8;
        *out_ptr.add(o_idx + 1) = (alpha * 255.0 + 0.5) as u8;

        // Shift both accumulators left
        let zero = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero));
        vst1q_f32(sp.add(4), vextq_f32::<1>(f1, zero));

        s_idx += 8;
        o_idx += 2;
    }
}

// --- NOGAMMA NEON ---

/// NEON horizontal upscale for RGB_NOGAMMA.
/// Like xscale_up_rgb but uses i2f (identity) instead of s2l (sRGB linearization).
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgb_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;

        smp_r = push_f_neon(smp_r, *i2f.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, *i2f.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, *i2f.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = vgetq_lane_f32::<0>(t2_r);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<0>(t2_g);
            *out_ptr.add(out_idx + 2) = vgetq_lane_f32::<0>(t2_b);
            *out_ptr.add(out_idx + 3) = vgetq_lane_f32::<1>(t2_r);
            *out_ptr.add(out_idx + 4) = vgetq_lane_f32::<1>(t2_g);
            *out_ptr.add(out_idx + 5) = vgetq_lane_f32::<1>(t2_b);

            out_idx += 6;
            coeff_idx += 8;
            j -= 2;
        }

        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);

            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

/// NEON horizontal upscale for RGBA_NOGAMMA (premultiplied alpha, no gamma).
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgba_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let mut smp_a = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha_new = *i2f.add(*in_ptr.add(in_base + 3) as usize);

        smp_a = push_f_neon(smp_a, alpha_new);
        smp_r = push_f_neon(smp_r, alpha_new * *i2f.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, alpha_new * *i2f.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, alpha_new * *i2f.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            let rg = vzip1q_f32(t2_r, t2_g);
            let ba = vzip1q_f32(t2_b, t2_a);
            vst1q_f32(out_ptr.add(out_idx), vcombine_f32(vget_low_f32(rg), vget_low_f32(ba)));
            vst1q_f32(out_ptr.add(out_idx + 4), vcombine_f32(vget_high_f32(rg), vget_high_f32(ba)));

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = dot4(smp_a, coeffs);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for RGBA_NOGAMMA.
/// Un-premultiplies, clamps, scales to 255, packs to bytes (no sRGB LUT).
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_rgba_nogamma(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 2 RGBA pixels (8 floats) at a time
    while i + 7 < len {
        let sum_a = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let sum_b = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));

        // Unpremultiply pixel 1
        let alpha_v = vdupq_laneq_f32::<3>(sum_a);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);
        let mut vals = sum_a;
        if alpha != 0.0 {
            vals = vmulq_f32(vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        // Replace alpha channel with clamped alpha
        let clamped = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v), clamped);
        let idx_a = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale), half));

        // Unpremultiply pixel 2
        let alpha_v2 = vdupq_laneq_f32::<3>(sum_b);
        let alpha_v2 = vminq_f32(vmaxq_f32(alpha_v2, zero), one);
        let alpha2 = vgetq_lane_f32::<0>(alpha_v2);
        let mut vals2 = sum_b;
        if alpha2 != 0.0 {
            vals2 = vmulq_f32(vals2, vdivq_f32(vdupq_n_f32(1.0), alpha_v2));
        }
        let clamped2 = vminq_f32(vmaxq_f32(vals2, zero), one);
        let clamped2 = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v2), clamped2);
        let idx_b = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped2, scale), half));

        // Pack both pixels to bytes and store 8 bytes
        let packed = vcombine_s16(vqmovn_s32(idx_a), vqmovn_s32(idx_b));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        vst1_u8(out_ptr.add(i), vget_low_u8(packed));

        i += 8;
    }

    // Remaining pixels one at a time
    while i < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let alpha_v = vdupq_laneq_f32::<3>(sum);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);
        let mut vals = sum;
        if alpha != 0.0 {
            vals = vmulq_f32(vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        let clamped = vminq_f32(vmaxq_f32(vals, zero), one);
        let clamped = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v), clamped);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale), half));
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        *(out_ptr.add(i) as *mut u32) = vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(packed)));

        i += 4;
    }
}

/// NEON horizontal upscale for RGBX_NOGAMMA.
/// Like xscale_up_rgbx but uses i2f (identity) instead of s2l (sRGB linearization).
#[target_feature(enable = "neon")]
pub unsafe fn xscale_up_rgbx_nogamma(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let mut smp_r = vdupq_n_f32(0.0);
    let mut smp_g = vdupq_n_f32(0.0);
    let mut smp_b = vdupq_n_f32(0.0);
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;

        smp_r = push_f_neon(smp_r, *i2f.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_neon(smp_g, *i2f.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_neon(smp_b, *i2f.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = vld1q_f32(coeff_ptr.add(coeff_idx));
            let c1 = vld1q_f32(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = vgetq_lane_f32::<0>(t2_r);
            *out_ptr.add(out_idx + 1) = vgetq_lane_f32::<0>(t2_g);
            *out_ptr.add(out_idx + 2) = vgetq_lane_f32::<0>(t2_b);
            *out_ptr.add(out_idx + 3) = 1.0;
            *out_ptr.add(out_idx + 4) = vgetq_lane_f32::<1>(t2_r);
            *out_ptr.add(out_idx + 5) = vgetq_lane_f32::<1>(t2_g);
            *out_ptr.add(out_idx + 6) = vgetq_lane_f32::<1>(t2_b);
            *out_ptr.add(out_idx + 7) = 1.0;

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = vld1q_f32(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = 1.0;

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// NEON vertical upscale for RGBX_NOGAMMA.
/// Clamps RGB to [0,1], scales to 255, X byte always 255 (no sRGB LUT).
#[target_feature(enable = "neon")]
pub unsafe fn yscale_up_rgbx_nogamma(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = vdupq_n_f32(coeffs[0]);
    let c1 = vdupq_n_f32(coeffs[1]);
    let c2 = vdupq_n_f32(coeffs[2]);
    let c3 = vdupq_n_f32(coeffs[3]);
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 2 RGBX pixels (8 floats) at a time
    while i + 7 < len {
        let sum_a = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let sum_b = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i + 4))),
            c2, vld1q_f32(l2.add(i + 4))),
            c1, vld1q_f32(l1.add(i + 4))),
            c0, vld1q_f32(l0.add(i + 4)));

        // Clamp to [0, 1], scale to 255, round
        let clamped_a = vminq_f32(vmaxq_f32(sum_a, zero), one);
        let idx_a = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped_a, scale), half));

        let clamped_b = vminq_f32(vmaxq_f32(sum_b, zero), one);
        let idx_b = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped_b, scale), half));

        // Pack both pixels to bytes
        let packed = vcombine_s16(vqmovn_s32(idx_a), vqmovn_s32(idx_b));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        vst1_u8(out_ptr.add(i), vget_low_u8(packed));

        // Overwrite X bytes with 255
        *out_ptr.add(i + 3) = 255;
        *out_ptr.add(i + 7) = 255;

        i += 8;
    }

    // Remaining pixels one at a time
    while i + 3 < len {
        let sum = vfmaq_f32(vfmaq_f32(vfmaq_f32(
            vmulq_f32(c3, vld1q_f32(l3.add(i))),
            c2, vld1q_f32(l2.add(i))),
            c1, vld1q_f32(l1.add(i))),
            c0, vld1q_f32(l0.add(i)));

        let clamped = vminq_f32(vmaxq_f32(sum, zero), one);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale), half));
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        *(out_ptr.add(i) as *mut u32) = vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(packed)));
        *out_ptr.add(i + 3) = 255;

        i += 4;
    }
}

/// NEON output for downscaled RGBA_NOGAMMA.
/// Un-premultiplies, clamps, scales to 255, packs to bytes (no sRGB LUT).
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_rgba_nogamma(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    let mut i = 0u32;

    // Process 2 pixels at a time
    while i + 1 < width {
        let sp = s_ptr.add(s_idx);

        // Pixel 1: load 4 accumulators
        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        let vals = gather_lane0(f0, f1, f2, f3);

        let alpha_v = vdupq_laneq_f32::<3>(vals);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);
        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = vmulq_f32(rgb_vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        rgb_vals = vminq_f32(vmaxq_f32(rgb_vals, zero), one);
        let rgb_vals = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v), rgb_vals);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(rgb_vals, scale), half));

        // Shift pixel 1 accumulators
        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero_v));

        // Pixel 2: load 4 accumulators
        let sp2 = s_ptr.add(s_idx + 16);
        let g0 = vld1q_f32(sp2);
        let g1 = vld1q_f32(sp2.add(4));
        let g2 = vld1q_f32(sp2.add(8));
        let g3 = vld1q_f32(sp2.add(12));

        let vals2 = gather_lane0(g0, g1, g2, g3);

        let alpha_v2 = vdupq_laneq_f32::<3>(vals2);
        let alpha_v2 = vminq_f32(vmaxq_f32(alpha_v2, zero), one);
        let alpha2 = vgetq_lane_f32::<0>(alpha_v2);
        let mut rgb_vals2 = vals2;
        if alpha2 != 0.0 {
            rgb_vals2 = vmulq_f32(rgb_vals2, vdivq_f32(vdupq_n_f32(1.0), alpha_v2));
        }
        rgb_vals2 = vminq_f32(vmaxq_f32(rgb_vals2, zero), one);
        let rgb_vals2 = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v2), rgb_vals2);
        let idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(rgb_vals2, scale), half));

        // Pack both pixels and store 8 bytes
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        vst1_u8(out_ptr.add(o_idx), vget_low_u8(packed));

        // Shift pixel 2 accumulators
        vst1q_f32(sp2,       vextq_f32::<1>(g0, zero_v));
        vst1q_f32(sp2.add(4),  vextq_f32::<1>(g1, zero_v));
        vst1q_f32(sp2.add(8),  vextq_f32::<1>(g2, zero_v));
        vst1q_f32(sp2.add(12), vextq_f32::<1>(g3, zero_v));

        s_idx += 32;
        o_idx += 8;
        i += 2;
    }

    // Remaining pixel
    while i < width {
        let sp = s_ptr.add(s_idx);

        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));
        let f3 = vld1q_f32(sp.add(12));

        let vals = gather_lane0(f0, f1, f2, f3);

        let alpha_v = vdupq_laneq_f32::<3>(vals);
        let alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
        let alpha = vgetq_lane_f32::<0>(alpha_v);
        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = vmulq_f32(rgb_vals, vdivq_f32(vdupq_n_f32(1.0), alpha_v));
        }
        rgb_vals = vminq_f32(vmaxq_f32(rgb_vals, zero), one);
        let rgb_vals = vsetq_lane_f32::<3>(vgetq_lane_f32::<0>(alpha_v), rgb_vals);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(rgb_vals, scale), half));
        let packed = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx));
        let packed = vcombine_u8(vqmovun_s16(packed), vqmovun_s16(packed));
        *(out_ptr.add(o_idx) as *mut u32) = vget_lane_u32::<0>(vreinterpret_u32_u8(vget_low_u8(packed)));

        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));
        vst1q_f32(sp.add(12), vextq_f32::<1>(f3, zero_v));

        s_idx += 16;
        o_idx += 4;
        i += 1;
    }
}

/// NEON output for downscaled RGBX_NOGAMMA.
/// Clamps RGB to [0,1], scales to 255, X byte always 255 (no sRGB LUT).
#[target_feature(enable = "neon")]
pub unsafe fn yscale_out_rgbx_nogamma(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let scale = vdupq_n_f32(255.0);
    let half = vdupq_n_f32(0.5);
    let zero = vdupq_n_f32(0.0);
    let one_f = vdupq_n_f32(1.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx);

        let f0 = vld1q_f32(sp);
        let f1 = vld1q_f32(sp.add(4));
        let f2 = vld1q_f32(sp.add(8));

        let vals = gather_lane0(f0, f1, f2, f2);

        // Clamp to [0, 1], scale to [0, 255], round
        let vals = vmaxq_f32(vals, zero);
        let vals = vminq_f32(vals, one_f);
        let idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale), half));

        *out_ptr.add(o_idx)     = vgetq_lane_s32::<0>(idx) as u8;
        *out_ptr.add(o_idx + 1) = vgetq_lane_s32::<1>(idx) as u8;
        *out_ptr.add(o_idx + 2) = vgetq_lane_s32::<2>(idx) as u8;
        *out_ptr.add(o_idx + 3) = 255;

        let zero_v = vdupq_n_f32(0.0);
        vst1q_f32(sp,       vextq_f32::<1>(f0, zero_v));
        vst1q_f32(sp.add(4),  vextq_f32::<1>(f1, zero_v));
        vst1q_f32(sp.add(8),  vextq_f32::<1>(f2, zero_v));

        s_idx += 16;
        o_idx += 4;
    }
}

/// NEON downscale for RGB_NOGAMMA: horizontal x-filtering + y-accumulation (no gamma).
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgb_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);

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
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2, s);

                in_idx += 6;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        }

        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
    }
}

/// NEON downscale for RGBX_NOGAMMA: horizontal x-filtering + y-accumulation (no gamma).
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgbx_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 2 {
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2, s);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx, s);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        sy = vfmaq_f32(sy, cy, sample);
        vst1q_f32(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Skip X channel y-accumulation
        sy_idx += 4;

        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
    }
}

/// NEON downscale for RGBA_NOGAMMA: horizontal x-filtering with premultiplied alpha + y-accumulation (no gamma).
#[target_feature(enable = "neon")]
pub unsafe fn scale_down_rgba_nogamma(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = vld1q_f32(coeffs_y.as_ptr());

    let mut sum_r = vdupq_n_f32(0.0);
    let mut sum_g = vdupq_n_f32(0.0);
    let mut sum_b = vdupq_n_f32(0.0);
    let mut sum_a = vdupq_n_f32(0.0);

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
            let mut sum_r2 = vdupq_n_f32(0.0);
            let mut sum_g2 = vdupq_n_f32(0.0);
            let mut sum_b2 = vdupq_n_f32(0.0);
            let mut sum_a2 = vdupq_n_f32(0.0);

            let mut j = 0;
            while j + 1 < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));
                let cx2 = vld1q_f32(cx_ptr.add(cx_idx + 4));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);
                sum_a = vaddq_f32(cx_a, sum_a);

                let cx2_a = vmulq_f32(cx2, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 7) as usize)));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = vfmaq_f32(sum_r2, cx2_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = vfmaq_f32(sum_g2, cx2_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = vfmaq_f32(sum_b2, cx2_a, s);
                sum_a2 = vaddq_f32(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);
                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = vaddq_f32(sum_r, sum_r2);
            sum_g = vaddq_f32(sum_g, sum_g2);
            sum_b = vaddq_f32(sum_b, sum_b2);
            sum_a = vaddq_f32(sum_a, sum_a2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = vld1q_f32(cx_ptr.add(cx_idx));

                let cx_a = vmulq_f32(cx, vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_r = vfmaq_f32(sum_r, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = vfmaq_f32(sum_g, cx_a, s);
                let s = vdupq_n_f32(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = vfmaq_f32(sum_b, cx_a, s);
                sum_a = vaddq_f32(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        let mut sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_r);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_g);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_b);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        sy = vld1q_f32(sy_ptr.add(sy_idx));
        let sample = vdupq_laneq_f32::<0>(sum_a);
        vst1q_f32(sy_ptr.add(sy_idx), vfmaq_f32(sy, cy, sample));
        sy_idx += 4;

        let zero = vdupq_n_f32(0.0);
        sum_r = vextq_f32::<1>(sum_r, zero);
        sum_g = vextq_f32::<1>(sum_g, zero);
        sum_b = vextq_f32::<1>(sum_b, zero);
        sum_a = vextq_f32::<1>(sum_a, zero);
    }
}

// --- Helpers ---

/// NEON push_f: shift left by one float, insert new value at position 3.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn push_f_neon(v: float32x4_t, val: f32) -> float32x4_t {
    let shifted = vextq_f32::<1>(v, vdupq_n_f32(0.0));
    vsetq_lane_f32::<3>(val, shifted)
}

/// Compute dot product of 4-element vector with one set of coefficients.
/// Returns the scalar result.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot4(smp: float32x4_t, coeffs: float32x4_t) -> f32 {
    let prod = vmulq_f32(smp, coeffs);
    let sum = vaddvq_f32(prod);
    sum
}

/// Compute two dot products simultaneously (smp * c0 and smp * c1).
/// Returns [dot0, dot1, ?, ?] in the float32x4_t result.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot4x2(smp: float32x4_t, c0: float32x4_t, c1: float32x4_t) -> float32x4_t {
    let p0 = vmulq_f32(smp, c0);
    let p1 = vmulq_f32(smp, c1);
    let lo = vzip1q_f32(p0, p1);
    let hi = vzip2q_f32(p0, p1);
    let sum = vaddq_f32(lo, hi);
    let t1 = vcombine_f32(vget_high_f32(sum), vget_high_f32(sum));
    vaddq_f32(sum, t1)
}

/// Gather lane 0 from 4 separate vectors into a single vector.
/// Returns [f0[0], f1[0], f2[0], f3[0]].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn gather_lane0(f0: float32x4_t, f1: float32x4_t, f2: float32x4_t, f3: float32x4_t) -> float32x4_t {
    let a01 = vzip1q_f32(f0, f1);
    let a23 = vzip1q_f32(f2, f3);
    vcombine_f32(vget_low_f32(a01), vget_low_f32(a23))
}
