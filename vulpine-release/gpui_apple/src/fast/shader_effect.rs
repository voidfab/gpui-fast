// Modified by Longbridge for gpui-fast.
// vulpine (FOX9-894bf2), not upstream: a shader pass over a rounded clip group.
//! A shader effect over any GPUI subtree, with no change to GPUI's scene.
//!
//! gpui-fast already gathers a subtree into a scene of its own: a rounded
//! `overflow_hidden` element's descendants become a rounded clip group, drawn
//! at the group's place in its parent's draw order (`rounded_clip::encode`).
//! vulpine's native kit (`native_kit::effect`) marks such a group by painting
//! sentinel quads first inside it. When the group's content has been drawn,
//! this pass copies what the target holds under the group's bounds (that
//! content over whatever lay beneath it), and draws one quad over those bounds
//! that reads the copy through the effect's fragment shader. Later draws
//! (siblings painted after, overlays) land on top, unshaded.
//!
//! The sentinel protocol (v1), shared with `native_kit::effect`:
//!
//! * a sentinel is a quad whose border color's hue is [`MAGIC`]; its border
//!   color's saturation is its slot divided by [`SLOTS`];
//! * each slot carries four values: the background's hue, saturation and
//!   lightness, then the border color's lightness. All sixteen are in 0..=1
//!   (so the published renderer, which does not know them, draws a quad whose
//!   color stays in range at an alpha it rounds to nothing);
//! * slot 0's quad is painted under the group's content mask, which is where
//!   the pass may draw (the group's sprite carries only its own shape);
//! * the sixteen values, in order: kind (0.25 ASCII, 0.5 CRT), mix, cell / 64
//!   (device px), tint, curve / 4, level, scanlines, (phase + 1) / 3, ground
//!   red, green, blue, threshold, curvature, aberration / 8 (device px),
//!   vignette, glow.
//!
//! A frame that cannot read its target (a framebuffer-only drawable), an empty
//! region or a pipeline that failed to build draws the group unshaded.

use std::cell::RefCell;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use block2::RcBlock;
use gpui::{Bounds, DevicePixels, ScaledPixels, Scene, Size};
use metal::{MTLOrigin, MTLSize};
use objc2::{msg_send, runtime::AnyObject};

use crate::metal_renderer::{MetalRenderer, new_command_encoder_for_texture};

/// The hue of a sentinel's border color: no color GPUI builds has it.
pub(crate) const MAGIC: f32 = -4093.0;
/// Slots of four values each.
pub(crate) const SLOTS: usize = 4;

const SOURCE: &str = include_str!("shader_effect.metal");

/// The uniforms the shader reads (`VulpineEffect` in shader_effect.metal).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Uniforms {
    region: [f32; 4],
    anchor: [f32; 4],
    look: [f32; 4],
    tone: [f32; 4],
    ground: [f32; 4],
    crt: [f32; 4],
    viewport: [f32; 2],
    pad: [f32; 2],
}

/// One group's effect, read from its scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Effect {
    values: [f32; SLOTS * 4],
    bounds: Bounds<ScaledPixels>,
    clip: Bounds<ScaledPixels>,
}

impl Effect {
    /// The effect a group's scene asks for: its four sentinels, or none (an
    /// ordinary rounded group, or slot 0 missing).
    pub(crate) fn find(scene: &Scene, bounds: Bounds<ScaledPixels>) -> Option<Self> {
        let mut values = [0f32; SLOTS * 4];
        let mut seen = [false; SLOTS];
        let mut clip = None;
        for quad in &scene.quads {
            let border = quad.border_color;
            if border.h != MAGIC {
                continue;
            }
            let slot = (border.s * SLOTS as f32).round();
            if !(0.0..SLOTS as f32).contains(&slot) {
                continue;
            }
            let Some(background) = quad.background.as_solid() else {
                continue;
            };
            let slot = slot as usize;
            values[slot * 4..slot * 4 + 4].copy_from_slice(&[
                background.h,
                background.s,
                background.l,
                border.l,
            ]);
            seen[slot] = true;
            if slot == 0 {
                clip = Some(quad.content_mask.bounds);
            }
        }
        let effect = Self {
            values,
            bounds,
            clip: clip?.intersect(&bounds),
        };
        (seen.iter().all(|s| *s) && effect.kind() != 0).then_some(effect)
    }

    /// 1 ASCII, 2 CRT, 0 none.
    fn kind(&self) -> u32 {
        (self.values[0] * 4.0).round() as u32
    }

    /// The target pixels the pass reads and writes: the visible part of the
    /// bounds, whole pixels, inside the target.
    fn region(&self, width: u64, height: u64) -> Option<(u64, u64, u64, u64)> {
        let clip = self.clip;
        let x0 = clip.origin.x.0.floor().max(0.0) as u64;
        let y0 = clip.origin.y.0.floor().max(0.0) as u64;
        let x1 = (clip.origin.x.0 + clip.size.width.0).ceil().max(0.0) as u64;
        let y1 = (clip.origin.y.0 + clip.size.height.0).ceil().max(0.0) as u64;
        let (x1, y1) = (x1.min(width), y1.min(height));
        (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
    }

    fn uniforms(&self, region: (u64, u64, u64, u64), viewport: Size<DevicePixels>) -> Uniforms {
        let v = &self.values;
        let b = self.bounds;
        Uniforms {
            region: [
                region.0 as f32,
                region.1 as f32,
                region.2 as f32,
                region.3 as f32,
            ],
            anchor: [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0],
            look: [self.kind() as f32, v[1], v[2] * 64.0, v[3]],
            tone: [v[4] * 4.0, v[5], v[6], v[7] * 3.0 - 1.0],
            ground: [v[8], v[9], v[10], v[11]],
            crt: [v[12], v[13] * 8.0, v[14], v[15]],
            viewport: [
                i32::from(viewport.width) as f32,
                i32::from(viewport.height) as f32,
            ],
            pad: [0.0; 2],
        }
    }

    /// Ends `encoder`, copies the region, and returns a new encoder on the
    /// same target (`rebind` restores the clip stack on it) with the effect's
    /// quad drawn. Any failure leaves the group as its content drew it.
    pub(crate) fn apply<'a>(
        &self,
        renderer: &MetalRenderer,
        texture: &'a metal::TextureRef,
        viewport: Size<DevicePixels>,
        commands: &'a metal::CommandBufferRef,
        encoder: &'a metal::RenderCommandEncoderRef,
        rebind: impl Fn(&metal::RenderCommandEncoderRef),
    ) -> &'a metal::RenderCommandEncoderRef {
        if texture.framebuffer_only() {
            warn_once("the target is framebuffer-only, so the effect cannot read it");
            return encoder;
        }
        let Some(region) = self.region(texture.width(), texture.height()) else {
            return encoder;
        };
        let Some(pipeline) = pipeline(&renderer.device, texture.pixel_format()) else {
            return encoder;
        };
        let (x, y, width, height) = region;
        // A level for every halving down to one texel: the ASCII pass samples
        // a cell's average at the level whose texels span one cell.
        let levels = 64 - width.max(height).leading_zeros() as u64;
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_pixel_format(texture.pixel_format());
        descriptor.set_mipmap_level_count(levels);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget);
        // Made for this frame and dropped once it is encoded: Metal keeps it
        // until the command buffer completes, so a window at rest holds none.
        let copy = renderer.device.new_texture(&descriptor);

        encoder.end_encoding();
        let blit = commands.new_blit_command_encoder();
        blit.copy_from_texture(
            texture,
            0,
            0,
            MTLOrigin { x, y, z: 0 },
            MTLSize {
                width,
                height,
                depth: 1,
            },
            &copy,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
        if levels > 1 {
            blit.generate_mipmaps(&copy);
        }
        blit.end_encoding();

        let encoder = new_command_encoder_for_texture(commands, texture, viewport, None);
        rebind(encoder);
        let uniforms = self.uniforms(region, viewport);
        let size = std::mem::size_of::<Uniforms>() as u64;
        let bytes = &uniforms as *const Uniforms as *const _;
        encoder.set_render_pipeline_state(&pipeline);
        encoder.set_vertex_bytes(0, size, bytes);
        encoder.set_fragment_bytes(0, size, bytes);
        encoder.set_fragment_texture(0, Some(&copy));
        encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        encoder
    }
}

/// The pass's pipeline, built once per device and pixel format from source
/// (the published shader library is left as it is).
#[derive(Default)]
struct Pipelines {
    built: Vec<(
        u64,
        metal::MTLPixelFormat,
        Option<metal::RenderPipelineState>,
    )>,
}

thread_local! {
    static PIPELINES: RefCell<Pipelines> = RefCell::new(Pipelines::default());
}

fn pipeline(
    device: &metal::DeviceRef,
    format: metal::MTLPixelFormat,
) -> Option<metal::RenderPipelineState> {
    let id = device.registry_id();
    PIPELINES.with(|cell| {
        let mut pipelines = cell.borrow_mut();
        if let Some((_, _, built)) = pipelines
            .built
            .iter()
            .find(|(d, f, _)| *d == id && *f == format)
        {
            return built.clone();
        }
        let built = build(device, format)
            .map_err(|error| warn_once(&format!("the effect pipeline did not build: {error}")))
            .ok();
        pipelines.built.push((id, format, built.clone()));
        built
    })
}

fn build(
    device: &metal::DeviceRef,
    format: metal::MTLPixelFormat,
) -> Result<metal::RenderPipelineState, String> {
    let library = device.new_library_with_source(SOURCE, &metal::CompileOptions::new())?;
    let vertex = library.get_function("vulpine_effect_vertex", None)?;
    let fragment = library.get_function("vulpine_effect_fragment", None)?;
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("vulpine_shader_effect");
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    let attachment = descriptor.color_attachments().object_at(0).unwrap();
    attachment.set_pixel_format(format);
    // The shader returns premultiplied color, already masked by the clip
    // stack's coverage: "over" with premultiplied alpha.
    attachment.set_blending_enabled(true);
    attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    device.new_render_pipeline_state(&descriptor)
}

fn warn_once(message: &str) {
    thread_local! {
        static WARNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if !WARNED.with(|w| w.replace(true)) {
        log::warn!("vulpine shader effect: {message}; the group draws unshaded");
    }
}

/// Frame GPU times in ms, kept only while `VULPINE_EFFECT_GPU_TIMING` is set (measurement runs).
static GPU_MS: Mutex<Vec<f64>> = Mutex::new(Vec::new());

fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VULPINE_EFFECT_GPU_TIMING").is_some())
}

/// With `VULPINE_EFFECT_GPU_TIMING` set, records the GPU time of a frame drawn through rounded clip groups (with or
/// without a look) when its command buffer completes: Metal's `GPUEndTime - GPUStartTime`.
pub(crate) fn time_frame(commands: &metal::CommandBufferRef) {
    if !timing() {
        return;
    }
    let block = RcBlock::new(|buffer: ptr::NonNull<AnyObject>| {
        // SAFETY: Metal hands its completion handlers the command buffer that completed; both are plain
        // CFTimeInterval reads of it.
        let buffer = unsafe { buffer.as_ref() };
        let start: f64 = unsafe { msg_send![buffer, GPUStartTime] };
        let end: f64 = unsafe { msg_send![buffer, GPUEndTime] };
        if let Ok(mut times) = GPU_MS.lock() {
            times.push((end - start) * 1000.0);
        }
    });
    // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI, as upstream's
    // `render_frame` relies on too.
    unsafe {
        commands.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
    }
}

/// The frame GPU times recorded since the last call (empty unless `VULPINE_EFFECT_GPU_TIMING` is set).
pub fn take_frame_gpu_ms() -> Vec<f64> {
    GPU_MS
        .lock()
        .map(|mut times| std::mem::take(&mut *times))
        .unwrap_or_default()
}
