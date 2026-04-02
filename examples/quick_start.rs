use oil_scale::{OilScale, ColorSpace};

fn main() {
    // Set up a scaler: 1920×1080 input → 640×360 output, RGB
    let mut scaler = OilScale::new(1920, 1080, 640, 360, ColorSpace::RGB).unwrap();
    let mut out_row = vec![0u8; 640 * 3];

    for _ in 0..360 {
        for _ in 0..scaler.slots() {
            let input_scanline: Vec<u8> = vec![0; 1920 * 3]; // read from your source
            scaler.push_scanline(&input_scanline);
        }
        scaler.read_scanline(&mut out_row);
    }

    println!("Scaled 1920x1080 -> 640x360 successfully");
}
