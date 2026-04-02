# CLAUDE.md

## Project overview

Oil is a Rust image resizing library (package name: `oil`, crate version 0.1.0). It is a port of the liboil C library focused on high-speed, low-memory, high-quality image resizing using Catmull-Rom cubic interpolation.

## Build and test commands

```sh
# Build (default, with libjpeg-turbo FFI)
cargo build --release

# Build without system dependencies
cargo build --release --no-default-features

# Run all tests
cargo test

# Run tests without FFI
cargo test --no-default-features

# Run specific test file
cargo test --test png_test
cargo test --test jpeg_test
cargo test --test reference

# Run imgscale CLI
cargo run --release --bin imgscale -- WIDTH HEIGHT input.jpg [output.jpg]

# Run benchmarks
cargo run --release --bin benchmark -- image.png
```

## Code structure

- `src/lib.rs` -- public API re-exports: `OilScale`, `OilError`, `ColorSpace`
- `src/scale.rs` -- core scaling engine (streaming scanline-by-scanline processing)
- `src/kernel.rs` -- Catmull-Rom kernel and coefficient computation
- `src/colorspace.rs` -- `ColorSpace` enum (G, GA, RGB, RGBA, ARGB, RGBX, CMYK)
- `src/srgb.rs` -- sRGB gamma conversion lookup tables
- `src/png.rs` -- PNG decode/resize/encode (`resize_png`, `resize_png_file`, `png_dimensions`)
- `src/jpeg.rs` -- pure Rust JPEG decode/resize/encode (`resize_jpeg`, `fix_ratio`)
- `src/jpeg_ffi.rs` -- libjpeg-turbo FFI bindings (feature-gated on `jpeg-turbo`)
- `src/sse2.rs` -- x86_64 SSE2 SIMD optimizations
- `src/bin/imgscale.rs` -- CLI image resizing tool
- `src/bin/benchmark.rs` -- performance benchmarking tool
- `csrc/jpeg_glue.c` -- C FFI glue for libjpeg-turbo
- `build.rs` -- compiles C FFI and links libjpeg when `jpeg-turbo` feature is enabled

## Key design decisions

- **Streaming API**: `OilScale` processes one scanline at a time to minimize memory. Call `slots()` to know how many input lines to `push_scanline()` before calling `read_scanline()`.
- **Separable convolution**: horizontal and vertical passes are independent, reducing complexity.
- **sRGB correctness**: all interpolation happens in linear light space using precomputed lookup tables.
- **Premultiplied alpha**: RGB channels are premultiplied by alpha before interpolation, then unpremultiplied on output.
- **Adaptive kernel width**: downscaling widens the kernel to cover all input pixels and prevent aliasing. Taps = `4 * (in_dim / out_dim)`, always even.
- **Dimension constraint**: both dimensions must scale in the same direction (both up or both down). Range: 1 to 1,000,000.

## Features (Cargo)

- `png` (default) -- PNG decode/resize/encode via `oil::png`
- `jpeg` (default) -- JPEG decode/resize/encode via `oil::jpeg`
- `jpeg-turbo` -- enables libjpeg-turbo FFI via `jpeg_ffi` module. Requires libjpeg-turbo dev headers on the system.
- Without default features -- core scaler only, no codec dependencies.

## Conventions

- JPEG quality in `imgscale` is hardcoded to 94.
- `fix_ratio()` lives in `jpeg.rs` but is used for all formats to preserve aspect ratio.
