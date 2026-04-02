# CLAUDE.md

## Build and test commands

```sh
# Build (default features: png + jpeg)
cargo build --release

# Build with libjpeg-turbo FFI (requires system headers)
cargo build --release --features jpeg-turbo

# Build core scaler only, no codec dependencies
cargo build --release --no-default-features

# Run all tests
cargo test

# Run tests without default features
cargo test --no-default-features

# Run imgscale CLI
cargo run --release --bin imgscale -- WIDTH HEIGHT input.jpg [output.jpg]

# Run benchmarks
cargo run --release --bin benchmark -- image.png
```

## Key design decisions

- **Streaming API**: `OilScale` processes one scanline at a time to minimize memory. Call `slots()` to know how many input lines to `push_scanline()` before calling `read_scanline()`.
- **Separable convolution**: horizontal and vertical passes are independent, reducing complexity.
- **sRGB correctness**: all interpolation happens in linear light space using precomputed lookup tables.
- **Premultiplied alpha**: RGB channels are premultiplied by alpha before interpolation, then unpremultiplied on output.
- **Adaptive kernel width**: downscaling widens the kernel to cover all input pixels and prevent aliasing. Taps = `4 * (in_dim / out_dim)`, always even.
- **Dimension constraint**: both dimensions must scale in the same direction (both up or both down). Range: 1 to 1,000,000.

## Conventions

- JPEG quality in `imgscale` is hardcoded to 94.
