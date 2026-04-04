# oil-scale

A high-speed, low-memory, high-quality image resizing library written in Rust.

Oil resizes images using Catmull-Rom cubic interpolation with proper sRGB gamma handling and premultiplied alpha. It processes images one scanline at a time, keeping memory usage minimal even for very large images.

This is a Rust port of the https://github.com/ender672/liboil C library.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oil-scale = "0.1"

# Or with libjpeg-turbo FFI. Lowest memory use, highest performance. Requires system headers:
# oil-scale = { version = "0.1", features = ["jpeg-turbo"] }

# Or with no default features (core scaler only — bring your own decoder/encoder):
# oil-scale = { version = "0.1", default-features = false }
```

### High-level API

Resize an entire PNG or JPEG in one call:

```rust
use oil_scale::png::resize_png;
use oil_scale::jpeg::resize_jpeg;
use oil_scale::fix_ratio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Resize a PNG to an exact size
    let input = std::fs::read("photo.png")?;
    let resized = resize_png(&input, 200, 150)?;
    std::fs::write("thumb.png", &resized)?;

    // Resize a JPEG (last argument is output quality 0–100)
    let input = std::fs::read("photo.jpg")?;
    let resized = resize_jpeg(&input, 200, 150, 90)?;
    std::fs::write("thumb.jpg", &resized)?;

    // Use fix_ratio to fit within a bounding box while preserving aspect ratio
    let (out_w, out_h) = fix_ratio(1920, 1080, 200, 200)?;
    // out_w=200, out_h=113
    let resized = resize_png(&std::fs::read("wide.png")?, out_w, out_h)?;
    std::fs::write("wide_thumb.png", &resized)?;

    Ok(())
}
```

### Low-level streaming API

Use `OilScale` directly when you have your own decoder/encoder. Feed input
scanlines one at a time and read output scanlines as they become available:

```rust
use oil_scale::{OilScale, ColorSpace};

fn downscale(input_pixels: &[u8]) -> Vec<u8> {
    let (in_w, in_h) = (1920, 1080);
    let (out_w, out_h) = (480, 270);
    let cs = ColorSpace::RGBA;
    let cmp = cs.components(); // 4

    let mut scaler = OilScale::new(in_w, in_h, out_w, out_h, cs).unwrap();

    let in_stride = in_w as usize * cmp;
    let out_stride = out_w as usize * cmp;
    let mut output = vec![0u8; out_h as usize * out_stride];

    let mut in_line = 0usize;
    for i in 0..out_h as usize {
        // push_scanline() until slots() returns 0
        while scaler.slots() > 0 {
            let scanline = &input_pixels[in_line * in_stride..(in_line + 1) * in_stride];
            scaler.push_scanline(scanline);
            in_line += 1;
        }
        scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]).unwrap();
    }
    output
}
```

## Features

- Catmull-Rom interpolation with correct sRGB gamma and premultiplied alpha
- Streaming scanline-by-scanline API; never loads the full image into memory when the decoder allows it
- SSE2 SIMD acceleration on x86_64

### Command-line tool

```
cargo run --release --bin imgscale -- WIDTH HEIGHT input.jpg [output.jpg]
```

Aspect ratio is preserved automatically. Output path defaults to `output.jpg` or `output.png` based on input format.

## Minimum Supported Rust Version

1.70

## Building

```sh
# Default build (png + jpeg, pure Rust)
cargo build --release

# Core scaler only, no codec dependencies
cargo build --release --no-default-features

# With libjpeg-turbo FFI support (requires system headers)
cargo build --release --features jpeg-turbo
```

## Running tests

```sh
cargo test
cargo test --no-default-features
```

## Benchmarking

Input must be an 8-bit RGBA PNG. Use an image at least 100x100 pixels — the benchmark scales down to 1% of the original size, so very small images will fail.

```sh
cargo run --release --bin benchmark -- image.png
cargo run --release --bin benchmark -- --down image.png    # downscale only
cargo run --release --bin benchmark -- --up image.png      # upscale only
cargo run --release --bin benchmark -- image.png RGB       # specific color space
```

Control iterations with the `OILITERATIONS` environment variable.

## License

MIT — see [LICENSE](LICENSE).
