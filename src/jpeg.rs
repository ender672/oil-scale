use crate::colorspace::ColorSpace;
use crate::scale::{Error, OilScale};

/// Decode a JPEG from bytes, resize it, and re-encode as JPEG.
pub fn resize_jpeg(
    input: &[u8],
    out_width: u32,
    out_height: u32,
    quality: u8,
) -> Result<Vec<u8>, Error> {
    let mut decoder = jpeg_decoder::Decoder::new(input);
    let pixels = decoder.decode().map_err(|e| Error::Codec(e.into()))?;
    let info = decoder.info().ok_or(Error::Codec("missing JPEG info".into()))?;

    let in_width = info.width as u32;
    let in_height = info.height as u32;

    let cs = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => ColorSpace::G,
        jpeg_decoder::PixelFormat::RGB24 => ColorSpace::RGB,
        jpeg_decoder::PixelFormat::CMYK32 => ColorSpace::CMYK,
        _ => return Err(Error::InvalidArgument),
    };
    let cmp = cs.components();

    let mut scaler = OilScale::new(in_width, in_height, out_width, out_height, cs)?;

    let in_stride = in_width as usize * cmp;
    let out_stride = out_width as usize * cmp;
    let mut output = vec![0u8; out_height as usize * out_stride];

    let mut in_line = 0usize;
    for i in 0..out_height as usize {
        while scaler.slots() > 0 {
            let row_start = in_line * in_stride;
            scaler.push_scanline(&pixels[row_start..row_start + in_stride])?;
            in_line += 1;
        }
        scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]);
    }

    let enc_color_type = match cs {
        ColorSpace::G => jpeg_encoder::ColorType::Luma,
        ColorSpace::RGB => jpeg_encoder::ColorType::Rgb,
        ColorSpace::CMYK => jpeg_encoder::ColorType::Cmyk,
        _ => return Err(Error::InvalidArgument),
    };

    let enc_width: u16 = out_width
        .try_into()
        .map_err(|_| Error::InvalidArgument)?;
    let enc_height: u16 = out_height
        .try_into()
        .map_err(|_| Error::InvalidArgument)?;

    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, quality);
    encoder
        .encode(&output, enc_width, enc_height, enc_color_type)
        .map_err(|e| Error::Codec(e.into()))?;

    Ok(buf)
}
