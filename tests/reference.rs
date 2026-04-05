use oil_scale::ColorSpace;
use oil_scale::{OilScale, Error, fix_ratio};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

static WORST: Mutex<f64> = Mutex::new(0.0);

fn srgb_to_linear_ref(in_f: f64) -> f64 {
    if in_f <= 0.0404482362771082 {
        in_f / 12.92
    } else {
        ((in_f + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_ref(in_f: f64) -> f64 {
    if in_f <= 0.0 {
        return 0.0;
    }
    if in_f >= 1.0 {
        return 1.0;
    }
    if in_f <= 0.00313066844250063 {
        in_f * 12.92
    } else {
        1.055 * in_f.powf(1.0 / 2.4) - 0.055
    }
}

fn cubic(b: f64, c: f64, x: f64) -> f64 {
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
            + (-18.0 + 12.0 * b + 6.0 * c) * x * x
            + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x * x * x
            + (6.0 * b + 30.0 * c) * x * x
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn ref_catrom(x: f64) -> f64 {
    cubic(0.0, 0.5, x)
}

fn ref_calc_coeffs(coeffs: &mut [f64], offset: f64, taps: usize, ltrim: usize, rtrim: usize) {
    assert!(taps - ltrim - rtrim > 0);
    let tap_mult = taps as f64 / 4.0;
    let mut fudge = 0.0;

    for (i, c) in coeffs.iter_mut().enumerate().take(taps) {
        if i < ltrim || i >= taps - rtrim {
            *c = 0.0;
            continue;
        }
        let tap_offset = 1.0 - offset - (taps as f64) / 2.0 + i as f64;
        *c = ref_catrom(tap_offset.abs() / tap_mult) / tap_mult;
        fudge += *c;
    }

    let mut total_check = 0.0;
    for c in coeffs.iter_mut().take(taps) {
        *c /= fudge;
        total_check += *c;
    }
    assert!(
        (total_check - 1.0).abs() < 0.0000000001,
        "coefficients don't sum to 1.0: {}",
        total_check
    );
}

fn calc_taps_check(dim_in: u32, dim_out: u32) -> usize {
    if dim_in < dim_out {
        4
    } else {
        let tmp = (dim_in as usize * 4) / dim_out as usize;
        tmp - (tmp % 2)
    }
}

fn ref_map(dim_in: u32, dim_out: u32, pos: u32) -> f64 {
    (pos as f64 + 0.5) * (dim_in as f64 / dim_out as f64) - 0.5
}

fn split_map_check(dim_in: u32, dim_out: u32, pos: u32) -> (i32, f64) {
    let smp = ref_map(dim_in, dim_out, pos);
    let smp_i = smp.floor() as i32;
    let ty = smp - smp_i as f64;
    (smp_i, ty)
}

fn clamp_f(val: f64) -> f64 {
    val.clamp(0.0, 1.0)
}

fn preprocess(pixel: &mut [f64], cs: ColorSpace) {
    match cs {
        ColorSpace::G | ColorSpace::CMYK | ColorSpace::RgbNoGamma => {}
        ColorSpace::GA => {
            pixel[0] *= pixel[1];
        }
        ColorSpace::RGB => {
            pixel[0] = srgb_to_linear_ref(pixel[0]);
            pixel[1] = srgb_to_linear_ref(pixel[1]);
            pixel[2] = srgb_to_linear_ref(pixel[2]);
        }
        ColorSpace::RGBA => {
            pixel[0] = pixel[3] * srgb_to_linear_ref(pixel[0]);
            pixel[1] = pixel[3] * srgb_to_linear_ref(pixel[1]);
            pixel[2] = pixel[3] * srgb_to_linear_ref(pixel[2]);
        }
        ColorSpace::RGBX => {
            pixel[0] = srgb_to_linear_ref(pixel[0]);
            pixel[1] = srgb_to_linear_ref(pixel[1]);
            pixel[2] = srgb_to_linear_ref(pixel[2]);
            pixel[3] = 1.0;
        }
        ColorSpace::RgbaNoGamma => {
            pixel[0] *= pixel[3];
            pixel[1] *= pixel[3];
            pixel[2] *= pixel[3];
        }
        ColorSpace::RgbxNoGamma => {
            pixel[3] = 1.0;
        }
        _ => {}
    }
}

fn postprocess(pixel: &mut [f64], cs: ColorSpace) {
    match cs {
        ColorSpace::G => {
            pixel[0] = clamp_f(pixel[0]);
        }
        ColorSpace::GA => {
            let alpha = clamp_f(pixel[1]);
            if alpha != 0.0 {
                pixel[0] /= alpha;
            }
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = alpha;
        }
        ColorSpace::RGB => {
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
        }
        ColorSpace::RGBA => {
            let alpha = clamp_f(pixel[3]);
            if alpha != 0.0 {
                pixel[0] /= alpha;
                pixel[1] /= alpha;
                pixel[2] /= alpha;
            }
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
            pixel[3] = alpha;
        }
        ColorSpace::CMYK => {
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = clamp_f(pixel[1]);
            pixel[2] = clamp_f(pixel[2]);
            pixel[3] = clamp_f(pixel[3]);
        }
        ColorSpace::RGBX => {
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
            pixel[3] = 1.0;
        }
        ColorSpace::RgbNoGamma => {
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = clamp_f(pixel[1]);
            pixel[2] = clamp_f(pixel[2]);
        }
        ColorSpace::RgbaNoGamma => {
            let alpha = clamp_f(pixel[3]);
            if alpha != 0.0 {
                pixel[0] /= alpha;
                pixel[1] /= alpha;
                pixel[2] /= alpha;
            }
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = clamp_f(pixel[1]);
            pixel[2] = clamp_f(pixel[2]);
            pixel[3] = alpha;
        }
        ColorSpace::RgbxNoGamma => {
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = clamp_f(pixel[1]);
            pixel[2] = clamp_f(pixel[2]);
            pixel[3] = 1.0;
        }
        _ => {}
    }
}

fn validate_scanline8(oil: &[u8], reference: &[f64], width: usize, cmp: usize) {
    for i in 0..width {
        for j in 0..cmp {
            let pos = i * cmp + j;
            let ref_f = reference[pos] * 255.0;
            let ref_i = ref_f.round() as i32;
            let error = (oil[pos] as f64 - ref_f).abs() - 0.5;

            {
                let mut worst = WORST.lock().unwrap();
                if error > *worst {
                    *worst = error;
                }
            }

            if error > 0.06 {
                panic!(
                    "[{}:{}] expected: {}, got {} ({:.9})",
                    i, j, ref_i, oil[pos], ref_f
                );
            }
        }
    }
}

fn ref_xscale(input: &[f64], in_width: usize, output: &mut [f64], out_width: usize, cmp: usize) {
    let taps = calc_taps_check(in_width as u32, out_width as u32);
    let mut coeffs = vec![0.0f64; taps];
    let max_pos = in_width as i32 - 1;

    for i in 0..out_width {
        let (smp_i, tx) = split_map_check(in_width as u32, out_width as u32, i as u32);
        let start = smp_i - (taps as i32 / 2 - 1);

        let start_safe = start.max(0);
        let ltrim = (start_safe - start) as usize;

        let mut taps_safe = taps - ltrim;
        if start_safe + taps_safe as i32 > max_pos {
            taps_safe = (max_pos - start_safe + 1) as usize;
        }
        let rtrim = ((start + taps as i32) - (start_safe + taps_safe as i32)) as usize;

        ref_calc_coeffs(&mut coeffs, tx, taps, ltrim, rtrim);

        for j in 0..cmp {
            output[i * cmp + j] = 0.0;
            for k in 0..taps_safe {
                let in_pos = (start_safe + k as i32) as usize;
                output[i * cmp + j] += coeffs[ltrim + k] * input[in_pos * cmp + j];
            }
        }
    }
}

fn ref_yscale(
    input: &[Vec<f64>],
    width: usize,
    in_height: usize,
    output: &mut [Vec<f64>],
    out_height: usize,
    cmp: usize,
) {
    let mut transposed = vec![0.0f64; in_height * cmp];
    let mut trans_scaled = vec![0.0f64; out_height * cmp];

    for i in 0..width {
        // Transpose column i
        for row in 0..in_height {
            for j in 0..cmp {
                transposed[row * cmp + j] = input[row][i * cmp + j];
            }
        }
        // Scale
        ref_xscale(&transposed, in_height, &mut trans_scaled, out_height, cmp);
        // Transpose back
        for row in 0..out_height {
            for j in 0..cmp {
                output[row][i * cmp + j] = trans_scaled[row * cmp + j];
            }
        }
    }
}

fn ref_scale(
    input: &[Vec<u8>],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    cs: ColorSpace,
) -> Vec<Vec<f64>> {
    let cmp = cs.components();
    let stride = cmp * in_width;

    // Horizontal scaling
    let mut intermediate: Vec<Vec<f64>> = Vec::with_capacity(in_height);
    let mut pre_line = vec![0.0f64; stride];

    for input_row in &input[..in_height] {
        // Convert chars to floats
        for j in 0..stride {
            pre_line[j] = input_row[j] as f64 / 255.0;
        }

        // Preprocess each pixel
        for j in 0..in_width {
            preprocess(&mut pre_line[j * cmp..(j + 1) * cmp], cs);
        }

        // Horizontal scale
        let mut out_row = vec![0.0f64; out_width * cmp];
        ref_xscale(&pre_line, in_width, &mut out_row, out_width, cmp);
        intermediate.push(out_row);
    }

    // Vertical scaling
    let mut output: Vec<Vec<f64>> = (0..out_height)
        .map(|_| vec![0.0f64; out_width * cmp])
        .collect();
    ref_yscale(&intermediate, out_width, in_height, &mut output, out_height, cmp);

    // Postprocess
    for out_row in &mut output[..out_height] {
        for j in 0..out_width {
            postprocess(&mut out_row[j * cmp..(j + 1) * cmp], cs);
        }
    }

    output
}

fn do_oil_scale(
    input: &[Vec<u8>],
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) -> Vec<Vec<u8>> {
    let cmp = cs.components();
    let mut os = OilScale::new(in_width, in_height, out_width, out_height, cs).unwrap();
    let mut output: Vec<Vec<u8>> = (0..out_height)
        .map(|_| vec![0u8; out_width as usize * cmp])
        .collect();

    let mut in_line = 0usize;
    for out_row in output.iter_mut().take(out_height as usize) {
        while os.slots() > 0 {
            os.push_scanline(&input[in_line]).unwrap();
            in_line += 1;
        }
        os.read_scanline(out_row).unwrap();
    }

    output
}

fn test_scale(
    in_width: u32,
    in_height: u32,
    input: &[Vec<u8>],
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) {
    let cmp = cs.components();

    let oil_output = do_oil_scale(input, in_width, in_height, out_width, out_height, cs);

    let ref_output = ref_scale(input, in_width as usize, in_height as usize, out_width as usize, out_height as usize, cs);

    for i in 0..out_height as usize {
        validate_scanline8(&oil_output[i], &ref_output[i], out_width as usize, cmp);
    }
}

fn test_scale_square_rand(rng: &mut StdRng, in_dim: u32, out_dim: u32, cs: ColorSpace) {
    let cmp = cs.components();
    let stride = cmp * in_dim as usize;

    let input: Vec<Vec<u8>> = (0..in_dim)
        .map(|_| {
            let mut row = vec![0u8; stride];
            for b in row.iter_mut() {
                *b = (rng.gen::<u32>() % 256) as u8;
            }
            row
        })
        .collect();

    test_scale(in_dim, in_dim, &input, out_dim, out_dim, cs);
}

fn test_scale_each_cs(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::G);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::GA);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGB);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGBA);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGBX);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::CMYK);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RgbNoGamma);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RgbaNoGamma);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RgbxNoGamma);
}

fn test_scale_all_permutations(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_scale_each_cs(rng, dim_a, dim_b);
    test_scale_each_cs(rng, dim_b, dim_a);
}

fn test_scale_catrom_extremes(cs: ColorSpace) {
    let cmp = cs.components();
    let mut input: Vec<Vec<u8>> = vec![vec![0u8; 4 * cmp]; 4];

    // Solid white center with black border, replicated across components
    for j in 0..cmp {
        input[1][cmp + j] = 255;
        input[1][2 * cmp + j] = 255;
        input[2][cmp + j] = 255;
        input[2][2 * cmp + j] = 255;
    }

    test_scale(4, 4, &input, 7, 7, cs);
}

#[test]
fn scale_5_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289551);
    test_scale_all_permutations(&mut rng, 5, 1);
}

#[test]
fn scale_8_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289552);
    test_scale_all_permutations(&mut rng, 8, 1);
}

#[test]
fn scale_8_to_3() {
    let mut rng = StdRng::seed_from_u64(1531289553);
    test_scale_all_permutations(&mut rng, 8, 3);
}

#[test]
fn scale_100_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289554);
    test_scale_all_permutations(&mut rng, 100, 1);
}

#[test]
fn scale_100_to_99() {
    let mut rng = StdRng::seed_from_u64(1531289555);
    test_scale_all_permutations(&mut rng, 100, 99);
}

#[test]
fn scale_2_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289556);
    test_scale_all_permutations(&mut rng, 2, 1);
}

fn do_oil_scale_with_reset(
    input: &[Vec<u8>],
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let cmp = cs.components();
    let mut os = OilScale::new(in_width, in_height, out_width, out_height, cs).unwrap();

    // First pass
    let mut output1: Vec<Vec<u8>> = (0..out_height)
        .map(|_| vec![0u8; out_width as usize * cmp])
        .collect();
    let mut in_line = 0usize;
    for out_row in output1.iter_mut().take(out_height as usize) {
        while os.slots() > 0 {
            os.push_scanline(&input[in_line]).unwrap();
            in_line += 1;
        }
        os.read_scanline(out_row).unwrap();
    }

    // Reset and do second pass with the same input
    os.reset();

    let mut output2: Vec<Vec<u8>> = (0..out_height)
        .map(|_| vec![0u8; out_width as usize * cmp])
        .collect();
    let mut in_line = 0usize;
    for out_row in output2.iter_mut().take(out_height as usize) {
        while os.slots() > 0 {
            os.push_scanline(&input[in_line]).unwrap();
            in_line += 1;
        }
        os.read_scanline(out_row).unwrap();
    }

    (output1, output2)
}

fn test_reset_square_rand(rng: &mut StdRng, in_dim: u32, out_dim: u32, cs: ColorSpace) {
    let cmp = cs.components();
    let stride = cmp * in_dim as usize;

    let input: Vec<Vec<u8>> = (0..in_dim)
        .map(|_| {
            let mut row = vec![0u8; stride];
            for b in row.iter_mut() {
                *b = (rng.gen::<u32>() % 256) as u8;
            }
            row
        })
        .collect();

    let (output1, output2) =
        do_oil_scale_with_reset(&input, in_dim, in_dim, out_dim, out_dim, cs);

    for i in 0..out_dim as usize {
        assert_eq!(
            output1[i], output2[i],
            "reset: row {} differs for {:?} {}->{}",
            i, cs, in_dim, out_dim
        );
    }
}

fn test_reset_each_cs(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::G);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::GA);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RGB);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RGBA);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RGBX);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::CMYK);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RgbNoGamma);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RgbaNoGamma);
    test_reset_square_rand(rng, dim_a, dim_b, ColorSpace::RgbxNoGamma);
}

#[test]
fn reset_downscale() {
    let mut rng = StdRng::seed_from_u64(1531289600);
    // Various downscale dimensions
    test_reset_each_cs(&mut rng, 8, 3);
    test_reset_each_cs(&mut rng, 100, 1);
    test_reset_each_cs(&mut rng, 100, 99);
    test_reset_each_cs(&mut rng, 5, 1);
}

#[test]
fn reset_upscale() {
    let mut rng = StdRng::seed_from_u64(1531289601);
    // Various upscale dimensions
    test_reset_each_cs(&mut rng, 1, 5);
    test_reset_each_cs(&mut rng, 3, 8);
    test_reset_each_cs(&mut rng, 1, 100);
    test_reset_each_cs(&mut rng, 99, 100);
}

#[test]
fn scale_catrom_extremes() {
    test_scale_catrom_extremes(ColorSpace::G);
    test_scale_catrom_extremes(ColorSpace::GA);
    test_scale_catrom_extremes(ColorSpace::RGB);
    test_scale_catrom_extremes(ColorSpace::RGBA);
    test_scale_catrom_extremes(ColorSpace::RGBX);
    test_scale_catrom_extremes(ColorSpace::CMYK);
    test_scale_catrom_extremes(ColorSpace::RgbNoGamma);
    test_scale_catrom_extremes(ColorSpace::RgbaNoGamma);
    test_scale_catrom_extremes(ColorSpace::RgbxNoGamma);
}

// --- discard_output_scanline tests ---

fn do_oil_scale_with_discard(
    input: &[Vec<u8>],
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) -> Vec<Vec<u8>> {
    let cmp = cs.components();
    let mut os = OilScale::new(in_width, in_height, out_width, out_height, cs).unwrap();
    let mut output: Vec<Vec<u8>> = (0..out_height)
        .map(|_| vec![0u8; out_width as usize * cmp])
        .collect();

    let mut in_line = 0usize;
    for out_row_idx in 0..out_height as usize {
        while os.slots() > 0 {
            os.push_scanline(&input[in_line]).unwrap();
            in_line += 1;
        }
        if out_row_idx % 2 == 1 {
            // Discard odd rows
            os.discard_output_scanline().unwrap();
        } else {
            os.read_scanline(&mut output[out_row_idx]).unwrap();
        }
    }

    output
}

fn test_discard_square_rand(rng: &mut StdRng, in_dim: u32, out_dim: u32, cs: ColorSpace) {
    let cmp = cs.components();
    let stride = cmp * in_dim as usize;

    let input: Vec<Vec<u8>> = (0..in_dim)
        .map(|_| {
            let mut row = vec![0u8; stride];
            for b in row.iter_mut() {
                *b = (rng.gen::<u32>() % 256) as u8;
            }
            row
        })
        .collect();

    // Get reference output (full scale, no discards)
    let reference = do_oil_scale(&input, in_dim, in_dim, out_dim, out_dim, cs);

    // Get output with discards on odd rows
    let discarded = do_oil_scale_with_discard(&input, in_dim, in_dim, out_dim, out_dim, cs);

    // Even rows must match exactly
    for i in (0..out_dim as usize).step_by(2) {
        assert_eq!(
            reference[i], discarded[i],
            "discard: row {} differs for {:?} {}->{}",
            i, cs, in_dim, out_dim
        );
    }
}

fn test_discard_each_cs(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::G);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::GA);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RGB);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RGBA);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RGBX);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::CMYK);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RgbNoGamma);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RgbaNoGamma);
    test_discard_square_rand(rng, dim_a, dim_b, ColorSpace::RgbxNoGamma);
}

#[test]
fn discard_downscale() {
    let mut rng = StdRng::seed_from_u64(1531289700);
    test_discard_each_cs(&mut rng, 100, 50);
    test_discard_each_cs(&mut rng, 8, 3);
}

#[test]
fn discard_upscale() {
    let mut rng = StdRng::seed_from_u64(1531289701);
    test_discard_each_cs(&mut rng, 50, 100);
    test_discard_each_cs(&mut rng, 3, 8);
}

#[test]
fn push_scanline_rejects_when_output_pending() {
    let mut os = OilScale::new(100, 100, 50, 50, ColorSpace::RGB).unwrap();
    let input = vec![0u8; 100 * 3];

    // Feed scanlines until an output is ready
    while os.slots() > 0 {
        os.push_scanline(&input).unwrap();
    }

    // Now slots() == 0, push_scanline must fail
    assert!(matches!(
        os.push_scanline(&input),
        Err(Error::InvalidArgument)
    ));
}

fn test_out_not_ready(in_dim: u32, out_dim: u32, cs: ColorSpace) {
    let cmp = cs.components();
    let out_stride = out_dim as usize * cmp;
    let mut buf = vec![0u8; out_stride];

    // Calling read_scanline/discard before any input should fail
    let mut os = OilScale::new(in_dim, in_dim, out_dim, out_dim, cs).unwrap();
    assert!(os.read_scanline(&mut buf).is_err());
    assert!(os.discard_output_scanline().is_err());

    // Feed one input line when more are needed — should still fail
    if os.slots() > 1 {
        let in_line = vec![0u8; in_dim as usize * cmp];
        os.push_scanline(&in_line).unwrap();
        assert!(os.slots() > 0);
        assert!(os.read_scanline(&mut buf).is_err());
        assert!(os.discard_output_scanline().is_err());
    }

    // Feed enough input — read_scanline should succeed
    let mut os = OilScale::new(in_dim, in_dim, out_dim, out_dim, cs).unwrap();
    while os.slots() > 0 {
        let in_line = vec![0u8; in_dim as usize * cmp];
        os.push_scanline(&in_line).unwrap();
    }
    assert!(os.read_scanline(&mut buf).is_ok());

    // Same but with discard
    let mut os = OilScale::new(in_dim, in_dim, out_dim, out_dim, cs).unwrap();
    while os.slots() > 0 {
        let in_line = vec![0u8; in_dim as usize * cmp];
        os.push_scanline(&in_line).unwrap();
    }
    assert!(os.discard_output_scanline().is_ok());
}

#[test]
fn out_not_ready_downscale() {
    for &cs in &[ColorSpace::G, ColorSpace::RGB, ColorSpace::RGBA, ColorSpace::CMYK, ColorSpace::GA,
                 ColorSpace::RgbNoGamma, ColorSpace::RgbaNoGamma, ColorSpace::RgbxNoGamma] {
        test_out_not_ready(100, 50, cs);
    }
}

#[test]
fn out_not_ready_upscale() {
    for &cs in &[ColorSpace::G, ColorSpace::RGB, ColorSpace::RGBA, ColorSpace::CMYK, ColorSpace::GA,
                 ColorSpace::RgbNoGamma, ColorSpace::RgbaNoGamma, ColorSpace::RgbxNoGamma] {
        test_out_not_ready(50, 100, cs);
    }
}

// --- Error path and validation tests ---

#[test]
fn new_rejects_zero_input_width() {
    assert!(matches!(
        OilScale::new(0, 100, 50, 50, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
}

#[test]
fn new_rejects_zero_input_height() {
    assert!(matches!(
        OilScale::new(100, 0, 50, 50, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
}

#[test]
fn new_rejects_zero_output_width() {
    assert!(matches!(
        OilScale::new(100, 100, 0, 50, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
}

#[test]
fn new_rejects_zero_output_height() {
    assert!(matches!(
        OilScale::new(100, 100, 50, 0, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
}

#[test]
fn new_rejects_exceeding_max_dimension() {
    let over = 1_000_001;
    assert!(OilScale::new(over, 100, 50, 50, ColorSpace::RGB).is_err());
    assert!(OilScale::new(100, over, 50, 50, ColorSpace::RGB).is_err());
    assert!(OilScale::new(100, 100, over, 50, ColorSpace::RGB).is_err());
    assert!(OilScale::new(100, 100, 50, over, ColorSpace::RGB).is_err());
}

#[test]
fn new_rejects_mismatched_scale_direction() {
    // Upscale width but downscale height
    assert!(matches!(
        OilScale::new(100, 100, 200, 50, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
    // Downscale width but upscale height
    assert!(matches!(
        OilScale::new(100, 100, 50, 200, ColorSpace::RGB),
        Err(Error::InvalidArgument)
    ));
}

#[test]
fn new_accepts_identity_scale() {
    assert!(OilScale::new(100, 100, 100, 100, ColorSpace::RGB).is_ok());
}

#[test]
fn new_accepts_max_dimension() {
    assert!(OilScale::new(1_000_000, 1_000_000, 500_000, 500_000, ColorSpace::G).is_ok());
}

// --- fix_ratio tests ---

#[test]
fn fix_ratio_landscape_fit() {
    // 1000x500 into 100x100 box → width-limited → (100, 50)
    let (w, h) = fix_ratio(1000, 500, 100, 100).unwrap();
    assert_eq!(w, 100);
    assert_eq!(h, 50);
}

#[test]
fn fix_ratio_portrait_fit() {
    // 500x1000 into 100x100 box → height-limited → (50, 100)
    let (w, h) = fix_ratio(500, 1000, 100, 100).unwrap();
    assert_eq!(w, 50);
    assert_eq!(h, 100);
}

#[test]
fn fix_ratio_square() {
    let (w, h) = fix_ratio(200, 200, 50, 50).unwrap();
    assert_eq!(w, 50);
    assert_eq!(h, 50);
}

#[test]
fn fix_ratio_extreme_aspect() {
    // Very wide: 10000x1 into 100x100 → (100, 1) since height rounds to minimum 1
    let (w, h) = fix_ratio(10000, 1, 100, 100).unwrap();
    assert_eq!(w, 100);
    assert!(h >= 1);
}

#[test]
fn fix_ratio_rejects_zero_dims() {
    assert!(fix_ratio(0, 100, 50, 50).is_err());
    assert!(fix_ratio(100, 0, 50, 50).is_err());
    assert!(fix_ratio(100, 100, 0, 50).is_err());
    assert!(fix_ratio(100, 100, 50, 0).is_err());
}

// --- ColorSpace::components tests ---

#[test]
fn colorspace_components() {
    assert_eq!(ColorSpace::G.components(), 1);
    assert_eq!(ColorSpace::GA.components(), 2);
    assert_eq!(ColorSpace::RGB.components(), 3);
    assert_eq!(ColorSpace::RGBA.components(), 4);
    assert_eq!(ColorSpace::RGBX.components(), 4);
    assert_eq!(ColorSpace::CMYK.components(), 4);
    assert_eq!(ColorSpace::RgbNoGamma.components(), 3);
    assert_eq!(ColorSpace::RgbaNoGamma.components(), 4);
    assert_eq!(ColorSpace::RgbxNoGamma.components(), 4);
}

// --- Error Display tests ---

#[test]
fn error_display() {
    let e = OilScale::new(0, 0, 0, 0, ColorSpace::RGB).unwrap_err();
    assert_eq!(format!("{e}"), "invalid argument");
}

#[test]
fn error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let e: Error = io_err.into();
    assert!(format!("{e}").contains("I/O error"));
    // Test std::error::Error::source()
    use std::error::Error as StdError;
    assert!(e.source().is_some());
}

// --- Accessor tests ---

#[test]
fn accessors_report_correct_dimensions() {
    let os = OilScale::new(1920, 1080, 640, 360, ColorSpace::RGBA).unwrap();
    assert_eq!(os.input_width(), 1920);
    assert_eq!(os.input_height(), 1080);
    assert_eq!(os.output_width(), 640);
    assert_eq!(os.output_height(), 360);
    assert_eq!(os.color_space(), ColorSpace::RGBA);
    assert!(!os.is_upscale());
}

#[test]
fn accessors_upscale() {
    let os = OilScale::new(100, 100, 200, 200, ColorSpace::G).unwrap();
    assert!(os.is_upscale());
}
