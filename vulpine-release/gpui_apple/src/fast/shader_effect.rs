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
use std::ptr::{self, NonNull};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use block2::RcBlock;
use gpui::{Bounds, DevicePixels, ScaledPixels, Scene, Size};
use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompileOptions, MTLDevice, MTLLibrary, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

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
    fn region(&self, width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
        let clip = self.clip;
        let x0 = clip.origin.x.0.floor().max(0.0) as usize;
        let y0 = clip.origin.y.0.floor().max(0.0) as usize;
        let x1 = (clip.origin.x.0 + clip.size.width.0).ceil().max(0.0) as usize;
        let y1 = (clip.origin.y.0 + clip.size.height.0).ceil().max(0.0) as usize;
        let (x1, y1) = (x1.min(width), y1.min(height));
        (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
    }

    fn uniforms(
        &self,
        region: (usize, usize, usize, usize),
        viewport: Size<DevicePixels>,
    ) -> Uniforms {
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
    /// quad drawn. Any failure before the copy leaves the group as its content
    /// drew it; past it, only a new encoder that cannot be made is an error.
    pub(crate) fn apply(
        &self,
        renderer: &MetalRenderer,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport: Size<DevicePixels>,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
        rebind: impl Fn(&ProtocolObject<dyn MTLRenderCommandEncoder>),
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
        if texture.isFramebufferOnly() {
            warn_once("the target is framebuffer-only, so the effect cannot read it");
            return Ok(encoder);
        }
        let Some(region) = self.region(texture.width(), texture.height()) else {
            return Ok(encoder);
        };
        let format = texture.pixelFormat();
        let Some(pipeline) = pipeline(&renderer.device, format) else {
            return Ok(encoder);
        };
        let (x, y, width, height) = region;
        // A level for every halving down to one texel: the ASCII pass samples
        // a cell's average at the level whose texels span one cell.
        let levels = (usize::BITS - width.max(height).leading_zeros()) as usize;
        let descriptor = MTLTextureDescriptor::new();
        // SAFETY: the region is non-empty and inside the target, so its size
        // and the level count it implies are valid for a 2D texture.
        unsafe {
            descriptor.setWidth(width);
            descriptor.setHeight(height);
            descriptor.setMipmapLevelCount(levels);
        }
        descriptor.setPixelFormat(format);
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
        // Made for this frame and dropped once it is encoded: Metal keeps it
        // until the command buffer completes, so a window at rest holds none.
        let Some(copy) = renderer.device.newTextureWithDescriptor(&descriptor) else {
            warn_once("the effect's copy texture could not be made");
            return Ok(encoder);
        };
        let Some(blit) = commands.blitCommandEncoder() else {
            warn_once("no blit encoder for the effect's copy");
            return Ok(encoder);
        };

        encoder.endEncoding();
        // SAFETY: the source region lies inside `texture` (see `region`), the
        // copy is that region's size at level 0, and both textures outlive the
        // command buffer's encoding.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
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
        }
        if levels > 1 {
            blit.generateMipmapsForTexture(&copy);
        }
        blit.endEncoding();

        let encoder = new_command_encoder_for_texture(commands, texture, viewport, None)?;
        rebind(&encoder);
        let uniforms = self.uniforms(region, viewport);
        let size = std::mem::size_of::<Uniforms>();
        let bytes = NonNull::from(&uniforms).cast();
        encoder.setRenderPipelineState(&pipeline);
        // SAFETY: `uniforms` is live for these calls, which copy its bytes into
        // the slots the effect's shaders read (VulpineEffect at buffer 0), and
        // the copy texture is retained by the encoder until the buffer completes.
        unsafe {
            encoder.setVertexBytes_length_atIndex(bytes, size, 0);
            encoder.setFragmentBytes_length_atIndex(bytes, size, 0);
            encoder.setFragmentTexture_atIndex(Some(&copy), 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 6);
        }
        Ok(encoder)
    }
}

/// The pass's pipeline, built once per device and pixel format from source
/// (the published shader library is left as it is).
type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

#[derive(Default)]
struct Pipelines {
    built: Vec<(u64, MTLPixelFormat, Option<Pipeline>)>,
}

thread_local! {
    static PIPELINES: RefCell<Pipelines> = RefCell::new(Pipelines::default());
}

fn pipeline(device: &ProtocolObject<dyn MTLDevice>, format: MTLPixelFormat) -> Option<Pipeline> {
    let id = device.registryID();
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

fn build(device: &ProtocolObject<dyn MTLDevice>, format: MTLPixelFormat) -> Result<Pipeline, String> {
    let describe = |error: Retained<objc2_foundation::NSError>| error.to_string();
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(SOURCE),
            Some(&MTLCompileOptions::new()),
        )
        .map_err(describe)?;
    let function = |name: &str| {
        library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| format!("no function {name} in the effect's library"))
    };
    let vertex = function("vulpine_effect_vertex")?;
    let fragment = function("vulpine_effect_fragment")?;
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str("vulpine_shader_effect")));
    descriptor.setVertexFunction(Some(&vertex));
    descriptor.setFragmentFunction(Some(&fragment));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setPixelFormat(format);
    // The shader returns premultiplied color, already masked by the clip
    // stack's coverage: "over" with premultiplied alpha.
    attachment.setBlendingEnabled(true);
    attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .map_err(describe)
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
pub(crate) fn time_frame(commands: &ProtocolObject<dyn MTLCommandBuffer>) {
    if !timing() {
        return;
    }
    let block = RcBlock::new(|buffer: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
        // SAFETY: Metal hands its completion handlers the command buffer that completed; both are plain
        // CFTimeInterval reads of it.
        let buffer = unsafe { buffer.as_ref() };
        let start: f64 = unsafe { msg_send![buffer, GPUStartTime] };
        let end: f64 = unsafe { msg_send![buffer, GPUEndTime] };
        if let Ok(mut times) = GPU_MS.lock() {
            times.push((end - start) * 1000.0);
        }
    });
    // SAFETY: Metal copies the live block and invokes it once after the frame completes, as upstream's
    // `render_frame` handler is; it only reads the buffer and locks GPU_MS.
    unsafe {
        commands.addCompletedHandler(RcBlock::as_ptr(&block));
    }
}

/// The frame GPU times recorded since the last call (empty unless `VULPINE_EFFECT_GPU_TIMING` is set).
pub fn take_frame_gpu_ms() -> Vec<f64> {
    GPU_MS
        .lock()
        .map(|mut times| std::mem::take(&mut *times))
        .unwrap_or_default()
}
