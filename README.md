# oil

A high-speed, low-memory, high-quality image resizing library written in Rust.

Oil resizes images using Catmull-Rom cubic interpolation with proper sRGB gamma handling and premultiplied alpha. It processes images one scanline at a time, keeping memory usage minimal even for very large images.

This is a Rust port of the https://github.com/ender672/liboil C library.

## Features

- Catmull-Rom interpolation with correct sRGB gamma and premultiplied alpha
- streaming scanline-by-scanline API; never loads the full image into memory when the decoder allows it
- SSE2 SIMD acceleration on x86_64

### Command-line tool

```
cargo run --release --bin imgscale -- WIDTH HEIGHT input.jpg [output.jpg]
```

Aspect ratio is preserved automatically. Output path defaults to `output.jpg` or `output.png` based on input format.

## Building

```sh
# Default build (includes libjpeg-turbo FFI support)
cargo build --release

# Without FFI (pure Rust, no system dependencies)
cargo build --release --no-default-features
```

## Running tests

```sh
cargo test
cargo test --no-default-features
```

## Benchmarking

Input must be an 8-bit RGBA PNG.

```sh
cargo run --release --bin benchmark -- image.png
cargo run --release --bin benchmark -- --down image.png    # downscale only
cargo run --release --bin benchmark -- --up image.png      # upscale only
cargo run --release --bin benchmark -- image.png RGB       # specific color space
```

Control iterations with the `OILITERATIONS` environment variable.
