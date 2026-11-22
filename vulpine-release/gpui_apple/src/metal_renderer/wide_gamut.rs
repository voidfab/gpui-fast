// Modified by Longbridge for gpui-fast.
// vulpine (FOX9-c83a7e, lane CG16): wide gamut for the Metal renderer.
//! Wide gamut: colors outside sRGB (oklch's and oklab's vivid ones) reach a P3 display.
//!
//! The published renderer draws into `BGRA8Unorm` with no colorspace on its layer: every channel is stored in 0..1,
//! so a color outside sRGB is clipped per channel before the window server sees it. In wide mode the drawable, every
//! pipeline's color attachment and every intermediate (the path texture, the layer tiles, the headless target) are
//! `RGBA16Float`, and the layer is tagged `kCGColorSpaceExtendedSRGB`: the values stay sRGB-encoded (gamma space,
//! so GPUI's blending is the same arithmetic), 0..1 is exactly today's sRGB, and values beyond it (a red of 1.08, a
//! green of -0.20) are the colors past sRGB that a P3 panel shows. `wantsExtendedDynamicRangeContent` stays off: P3 is
//! a wider gamut, not a brighter one.
//!
//! The switch is per process ([`set_wide_gamut`] for windows, [`set_headless_wide_gamut`] for headless renderers,
//! both off by default, the published behavior), and each renderer follows it on its own next frame: it re-tags its
//! layer and rebuilds its pipelines and intermediates for the new format. Nothing runs at rest; a caller that flips
//! the switch asks its windows for a frame.
//!
//! GPUI's colors already carry values beyond sRGB: `Hsla` built from an `Rgba` whose channels lie outside 0..1 keeps
//! them (its saturation above 1, its lightness outside 0..1), and the shaders' `hsla_to_rgba` clamps nothing. Only
//! `Rgba::from(Hsla)` and the `hsla()` constructor clamp, on the CPU, and the renderer never calls them. The shaders'
//! gamma curves are made sign-preserving so an Oklab gradient between extended colors stays defined.

use super::{
    MetalRenderer, build_path_rasterization_pipeline_state, build_path_sprite_pipeline_state,
    build_pipeline_state,
};
use anyhow::{Context as _, Result};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceExtendedSRGB};
use objc2_metal::{MTLDevice as _, MTLPixelFormat, MTLTexture as _};
use objc2_quartz_core::CAMetalLayer;
use std::sync::atomic::{AtomicBool, Ordering};

/// The format of a wide-gamut renderer's drawable, pipelines and intermediates.
pub const WIDE_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
/// The published format (8-bit sRGB values, no colorspace on the layer).
pub const NARROW_FORMAT: MTLPixelFormat = MTLPixelFormat::BGRA8Unorm;

static WINDOWS: AtomicBool = AtomicBool::new(false);
static HEADLESS: AtomicBool = AtomicBool::new(false);

/// Asks every layer-backed renderer (a window's, its overlays') for wide gamut, or for the published 8-bit sRGB.
/// Off by default. A renderer opened after the call starts in that mode; an open one switches on its next frame.
pub fn set_wide_gamut(on: bool) {
    WINDOWS.store(on, Ordering::Relaxed);
}

/// Whether layer-backed renderers are asked for wide gamut.
pub fn wide_gamut() -> bool {
    WINDOWS.load(Ordering::Relaxed)
}

/// The same switch for headless renderers (visual tests, readbacks), apart from the windows' so a gate's baselines
/// stay 8-bit unless a test asks. Off by default.
pub fn set_headless_wide_gamut(on: bool) {
    HEADLESS.store(on, Ordering::Relaxed);
}

/// The format a switch asks for.
fn format_for(on: bool) -> MTLPixelFormat {
    if on { WIDE_FORMAT } else { NARROW_FORMAT }
}

/// The format a renderer opens with: its layer's (set by `configure_layer_format`), or the headless switch's.
pub(super) fn initial_format(layer: Option<&CAMetalLayer>) -> MTLPixelFormat {
    match layer {
        Some(layer) => layer.pixelFormat(),
        None => format_for(HEADLESS.load(Ordering::Relaxed)),
    }
}

/// Sets a new layer's pixel format and colorspace from the windows' switch (`MetalRenderer::configure_layer`).
pub(super) fn configure_layer_format(layer: &CAMetalLayer) {
    tag_layer(layer, format_for(wide_gamut()));
}

/// A layer's pixel format and colorspace for `format`: extended sRGB for the wide format, none (the published
/// layer's) otherwise.
fn tag_layer(layer: &CAMetalLayer, format: MTLPixelFormat) {
    layer.setPixelFormat(format);
    let space = (format == WIDE_FORMAT)
        .then(|| {
            // SAFETY: a CoreGraphics constant, valid for the process's life.
            CGColorSpace::with_name(Some(unsafe { kCGColorSpaceExtendedSRGB }))
        })
        .flatten();
    // `colorspace` is a retained property: the layer keeps its own reference.
    layer.setColorspace(space.as_deref());
}

/// Whether a layer is tagged extended sRGB (its colorspace's name), for tests and receipts.
pub fn layer_is_extended_srgb(layer: &CAMetalLayer) -> bool {
    let Some(space) = layer.colorspace() else {
        return false;
    };
    let Some(name) = CGColorSpace::name(Some(&space)) else {
        return false;
    };
    // SAFETY: a CoreGraphics constant, valid for the process's life.
    *name == *unsafe { kCGColorSpaceExtendedSRGB }
}

impl MetalRenderer {
    /// The pixel format this renderer draws in (the drawable's, every pipeline's, every intermediate's).
    pub fn pixel_format(&self) -> MTLPixelFormat {
        self.pixel_format
    }

    /// Whether this renderer draws in wide gamut now.
    pub fn is_wide_gamut(&self) -> bool {
        self.pixel_format == WIDE_FORMAT
    }

    /// Follows the switch that governs this renderer (the windows' with a layer, the headless one without): when it
    /// asks for another format, re-tags the layer and rebuilds the pipelines and intermediates. One atomic load a
    /// frame when nothing changed.
    pub(crate) fn follow_wide_gamut(&mut self) {
        let on = if self.layer.is_some() {
            wide_gamut()
        } else {
            HEADLESS.load(Ordering::Relaxed)
        };
        let format = format_for(on);
        if format != self.pixel_format
            && let Err(error) = self.switch_pixel_format(format)
        {
            // The renderer keeps the format it had, whole: its pipelines are built before any is replaced.
            log::error!("wide gamut: could not switch to {format:?}: {error:#}");
        }
    }

    fn switch_pixel_format(&mut self, format: MTLPixelFormat) -> Result<()> {
        #[cfg(feature = "runtime_shaders")]
        let library = self
            .device
            .newLibraryWithSource_options_error(
                &objc2_foundation::NSString::from_str(super::SHADERS_SOURCE_FILE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .context("building Metal library")?;
        #[cfg(not(feature = "runtime_shaders"))]
        let library = self
            .device
            .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(
                super::SHADERS_METALLIB,
            ))
            .context("loading Metal library")?;
        let device = &self.device;
        let pipeline = |label, vertex, fragment| {
            build_pipeline_state(device, &library, label, vertex, fragment, format)
        };
        let shadows = pipeline("shadows", "shadow_vertex", "shadow_fragment")?;
        let quads = pipeline("quads", "quad_vertex", "quad_fragment")?;
        let underlines = pipeline("underlines", "underline_vertex", "underline_fragment")?;
        let monochrome = pipeline(
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
        )?;
        let polychrome = pipeline(
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
        )?;
        let surfaces = pipeline("surfaces", "surface_vertex", "surface_fragment")?;
        let surfaces_bgra = pipeline("surfaces_bgra", "surface_vertex", "surface_bgra_fragment")?;
        let path_sprites = build_path_sprite_pipeline_state(
            device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            format,
        )?;
        let paths_rasterization = build_path_rasterization_pipeline_state(
            device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            format,
            self.path_sample_count,
        )?;
        if let Some(layer) = &self.layer {
            tag_layer(layer, format);
        }
        self.shadows_pipeline_state = shadows;
        self.quads_pipeline_state = quads;
        self.underlines_pipeline_state = underlines;
        self.monochrome_sprites_pipeline_state = monochrome;
        self.polychrome_sprites_pipeline_state = polychrome;
        self.surfaces_pipeline_state = surfaces;
        self.surfaces_bgra_pipeline_state = surfaces_bgra;
        self.path_sprites_pipeline_state = path_sprites;
        self.paths_rasterization_pipeline_state = paths_rasterization;
        self.pixel_format = format;
        // The path intermediates and the layer tiles are render targets of the old format. The intermediates are
        // kept while their size holds, so they are dropped first.
        if let Some(texture) = self.path_intermediate_texture.take() {
            self.path_intermediate_msaa_texture = None;
            let size = gpui::size(
                gpui::DevicePixels(texture.width() as i32),
                gpui::DevicePixels(texture.height() as i32),
            );
            self.update_path_intermediate_textures(size)?;
        }
        crate::fast::layers::TileCache::clear(&mut self.fast_layers);
        #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
        {
            self.headless_render_target = None;
        }
        Ok(())
    }
}

/// A frame read back from a wide-gamut target: linear rows of RGBA, the stored (sRGB-encoded, extended) values.
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct WideReadback {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl WideReadback {
    /// The stored value at (x, y).
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        self.pixels[(y * self.width + x) as usize]
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
static LAST_READBACK: parking_lot::Mutex<Option<WideReadback>> = parking_lot::Mutex::new(None);

/// The last frame a headless wide-gamut renderer read back (`render_scene_to_image` hands GPUI an 8-bit picture,
/// clipped to sRGB; this keeps the values it clipped). Taken once.
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub fn take_wide_readback() -> Option<WideReadback> {
    LAST_READBACK.lock().take()
}

/// Reads an `RGBA16Float` texture: keeps its values for [`take_wide_readback`] and returns them clipped to 8-bit
/// sRGB, as the published readback returns its BGRA8 target.
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub(super) fn read_wide_texture(
    texture: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>,
) -> anyhow::Result<image::RgbaImage> {
    use objc2_metal::{MTLOrigin, MTLRegion, MTLSize};
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 8;
    let mut halves = vec![0u16; height as usize * width as usize * 4];
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    // SAFETY: `halves` holds the region's rows of four halves each, `bytes_per_row` apart, and the texture is an
    // RGBA16Float one the CPU can read (shared or synchronized managed storage, as `render_offscreen` makes).
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            std::ptr::NonNull::from(halves.as_mut_slice()).cast(),
            bytes_per_row,
            region,
            0,
        );
    }
    let pixels: Vec<[f32; 4]> = halves
        .chunks_exact(4)
        .map(|p| [half(p[0]), half(p[1]), half(p[2]), half(p[3])])
        .collect();
    let bytes = pixels
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect();
    *LAST_READBACK.lock() = Some(WideReadback {
        width,
        height,
        pixels,
    });
    image::RgbaImage::from_raw(width, height, bytes)
        .context("failed to create RgbaImage from pixel data")
}

/// An IEEE 754 half as an f32.
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn half(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // A subnormal: normalize it.
            let mut e = 113u32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::half;

    #[test]
    fn halves_read_as_their_values() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xbc00), -1.0);
        assert_eq!(half(0x0000), 0.0);
        assert_eq!(half(0x3800), 0.5);
        assert_eq!(half(0x4000), 2.0);
        assert!((half(0x3c66) - 1.0996094).abs() < 1e-6);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
    }
}
