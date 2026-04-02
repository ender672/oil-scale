# oil-scale

A high-speed, low-memory, high-quality image resizing library written in Rust.

Oil resizes images using Catmull-Rom cubic interpolation with proper sRGB gamma handling and premultiplied alpha. It processes images one scanline at a time, keeping memory usage minimal even for very large images.

This is a Rust port of the https://github.com/ender672/liboil C library.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oil-scale = "0.1"
```

### High-level API

Resize an entire PNG or JPEG in one call:

```rust
use oil_scale::png::resize_png;

let input = std::fs::read("input.png").unwrap();
let resized = resize_png(&input, 200, 150).unwrap();
std::fs::write("output.png", &resized).unwrap();
```

```rust
use oil_scale::jpeg::resize_jpeg;

let input = std::fs::read("input.jpg").unwrap();
let resized = resize_jpeg(&input, 200, 150, 90).unwrap();
std::fs::write("output.jpg", &resized).unwrap();
```

Use `fix_ratio` to preserve aspect ratio within a bounding box:

```rust
use oil_scale::fix_ratio;

let (out_w, out_h) = fix_ratio(1920, 1080, 200, 200).unwrap();
// out_w=200, out_h=113
```

### Low-level streaming API

Use `OilScale` directly when you have your own decoder/encoder. Feed input scanlines one at a time and read output scanlines as they become available:

```rust
use oil_scale::{OilScale, ColorSpace};

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
    scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]);
}
```

## Features

- Catmull-Rom interpolation with correct sRGB gamma and premultiplied alpha
- Streaming scanline-by-scanline API; never loads the full image into memory when the decoder allows it
- SSE2 SIMD acceleration on x86_64

### Cargo features

| Feature | Default | Description |
|---|---|---|
| `png` | yes | PNG decode/resize/encode via `oil_scale::png` |
| `jpeg` | yes | JPEG decode/resize/encode via `oil_scale::jpeg` |
| `jpeg-turbo` | no | libjpeg-turbo FFI via `oil_scale::jpeg_ffi` (requires system headers) |

With no features enabled, you get the core scaler (`OilScale`, `ColorSpace`, `fix_ratio`) and no codec dependencies.

```toml
# Default (png + jpeg adapters)
oil-scale = "0.1"

# Core scaler only — bring your own decoder/encoder
oil-scale = { version = "0.1", default-features = false }

# Just PNG support
oil-scale = { version = "0.1", default-features = false, features = ["png"] }

# Everything including FFI
oil-scale = { version = "0.1", features = ["jpeg-turbo"] }
```

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
