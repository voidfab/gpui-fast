// vulpine (FOX9-a010dd), not upstream: CSS `backdrop-filter: blur()`, ported from Skathi's gpui fork
// (Apache-2.0: crates/gpui_apple/src/metal_renderer/backdrop.rs and shaders.metal; see the repository's NOTICE).
//! Backdrop blur: frosted glass over what a frame has drawn so far.
//!
//! A [`gpui::BackdropBlurRect`] batch breaks the frame's render pass. The pixels the batch's rects can read (their
//! union, grown by the kernel's reach) are copied from the frame target into a snapshot, blurred through a halving
//! Dual-Kawase pyramid (down: the centre at one half and four diagonal taps at one eighth; up: eight taps at one
//! eighth), and each rect is composited with blending off: inside its rounded rect the blurred copy (tinted), outside
//! it the untouched snapshot, so the antialiased corner stays exact. The frame then continues in a new pass on the
//! same target.
//!
//! What it costs and when:
//!
//! * A frame without a blur pays nothing: no pipeline is built, no texture is allocated, the drawable is untouched
//!   ([`prepare_frame`] returns the drawable as the target).
//! * The pyramid's levels are allocated up to the deepest a frame asks for (5 at most, [`gpui::MAX_BACKDROP_BLUR_KERNEL_LEVELS`]),
//!   and every scratch texture is released after [`SCRATCH_RELEASE_FRAMES`] frames that do not need it.
//! * A drawable that is framebuffer-only (every shipped window: only `vulpine-effect` and debug test builds make the
//!   layer readable) cannot be copied from: such a frame is drawn into an intermediate texture and copied onto the
//!   drawable at its end, and the layer's `framebufferOnly` is lifted for the drawables vended after it, until the
//!   scratch is released, when it is set again. This is the one readable-target lift; the shader effect
//!   (`fast::shader_effect`), which copies from the target too, reads the same intermediate when a blur shares its frame.
//! * Nothing here schedules a frame: a static frosted card draws once and rests like any other primitive.

use std::{mem, ops::Range, ptr};

use anyhow::{Context as _, Result};
use gpui::{
    BackdropBlurRect, Bounds, DevicePixels, MAX_BACKDROP_BLUR_KERNEL_LEVELS, ScaledPixels, Scene,
    Size,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLDevice,
    MTLFunction, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResource, MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

use crate::metal_renderer::{InstanceBinding, MetalRenderer, new_command_encoder_for_texture};

/// Frames the scratch textures may sit unused before they are released (Skathi's D3D12 rule): making one again is a
/// window-sized allocation, so short gaps keep them.
pub(crate) const SCRATCH_RELEASE_FRAMES: u32 = 90;

const LEVELS: usize = MAX_BACKDROP_BLUR_KERNEL_LEVELS as usize;

/// The shaders' bindings (shaders.metal, "vulpine (FOX9-a010dd)").
const INSTANCES_INDEX: usize = 0;
const VIEWPORT_INDEX: usize = 1;
const PROCESSED_TEXTURE: usize = 0;
const ORIGINAL_TEXTURE: usize = 1;

pub(crate) type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
pub(crate) type Encoder = Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>;

/// The renderer's backdrop state: the library its pipelines come from, the pipelines once a frame has needed them,
/// the scratch textures and the layer flag it lifted.
pub(crate) struct Backdrop {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Option<Pipelines>,
    /// The pipelines failed to build for this format: blurs are skipped (announced once).
    unavailable: Option<MTLPixelFormat>,
    targets: Option<Targets>,
    /// Consecutive frames that held more scratch than they needed.
    idle_frames: u32,
    /// This renderer set its layer's `framebufferOnly` to false, and sets it back when the scratch goes.
    framebuffer_only_lifted: bool,
    /// Whether the frame being encoded draws its blurs (its target can be read and the pipelines exist).
    frame_enabled: bool,
}

struct Pipelines {
    format: MTLPixelFormat,
    downsample: Pipeline,
    upsample: Pipeline,
    frame_copy: Pipeline,
    composite: Pipeline,
}

/// Sized to the frame.
struct Targets {
    width: usize,
    height: usize,
    format: MTLPixelFormat,
    snapshot: Texture,
    /// Level `i` is the frame halved `i + 1` times, allocated up to the deepest level a frame has asked for.
    levels: Vec<Texture>,
    /// The intermediate a framebuffer-only drawable's frame is drawn into.
    frame: Option<Texture>,
}

/// Where a frame renders: the drawable, or an intermediate copied onto it at the end.
pub(crate) struct FrameTarget {
    texture: Texture,
    copy_to: Option<Texture>,
}

impl FrameTarget {
    /// The texture the frame's passes draw into.
    pub(crate) fn texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }
}

impl Backdrop {
    pub(crate) fn new(library: &ProtocolObject<dyn MTLLibrary>) -> Self {
        Self {
            library: library.retain(),
            pipelines: None,
            unavailable: None,
            targets: None,
            idle_frames: 0,
            framebuffer_only_lifted: false,
            frame_enabled: false,
        }
    }

    /// The bytes the scratch textures hold now (Metal's `allocatedSize`), for measurement runs and tests.
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.targets.as_ref().map_or(0, |targets| {
            let textures = std::iter::once(&targets.snapshot)
                .chain(&targets.levels)
                .chain(&targets.frame);
            textures
                .map(|texture| MTLAllocation::allocatedSize(&**texture))
                .sum()
        })
    }

    #[cfg(test)]
    pub(crate) fn held_levels(&self) -> Option<usize> {
        self.targets.as_ref().map(|targets| targets.levels.len())
    }

    #[cfg(test)]
    pub(crate) fn has_pipelines(&self) -> bool {
        self.pipelines.is_some()
    }

    /// Keeps the scratch a frame needs (`needed` counts the snapshot as one more level), releasing the rest after
    /// [`SCRATCH_RELEASE_FRAMES`] frames in a row that held more.
    fn trim(&mut self, needed: usize) {
        let held = self
            .targets
            .as_ref()
            .map_or(0, |targets| targets.levels.len() + 1);
        if needed >= held {
            self.idle_frames = 0;
            return;
        }
        self.idle_frames += 1;
        if self.idle_frames < SCRATCH_RELEASE_FRAMES {
            return;
        }
        self.idle_frames = 0;
        match (needed, &mut self.targets) {
            (0, targets) => *targets = None,
            (needed, Some(targets)) => targets.levels.truncate(needed - 1),
            (_, None) => {}
        }
    }

    fn ensure_pipelines(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        format: MTLPixelFormat,
    ) -> bool {
        if self.pipelines.as_ref().is_some_and(|p| p.format == format) {
            return true;
        }
        if self.unavailable == Some(format) {
            return false;
        }
        match Pipelines::new(device, &self.library, format) {
            Ok(pipelines) => {
                self.pipelines = Some(pipelines);
                true
            }
            Err(error) => {
                log::warn!(
                    "backdrop blur is unavailable: its pipelines failed to build: {error:#}"
                );
                self.pipelines = None;
                self.unavailable = Some(format);
                false
            }
        }
    }

    fn ensure_targets(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        width: usize,
        height: usize,
        format: MTLPixelFormat,
        levels: usize,
    ) -> Result<&mut Targets> {
        let targets = match self.targets.take().filter(|targets| {
            targets.width == width && targets.height == height && targets.format == format
        }) {
            Some(targets) => targets,
            None => Targets {
                width,
                height,
                format,
                snapshot: texture(
                    device,
                    "backdrop_snapshot",
                    width,
                    height,
                    format,
                    MTLTextureUsage::ShaderRead,
                )?,
                levels: Vec::new(),
                frame: None,
            },
        };
        let targets = self.targets.insert(targets);
        while targets.levels.len() < levels {
            let divisor = 1usize << (targets.levels.len() + 1);
            targets.levels.push(texture(
                device,
                "backdrop_blur_level",
                (width / divisor).max(1),
                (height / divisor).max(1),
                format,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            )?);
        }
        Ok(targets)
    }
}

impl Pipelines {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        format: MTLPixelFormat,
    ) -> Result<Self> {
        let function = |name: &str| -> Result<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .with_context(|| format!("locating function {name}"))
        };
        let pass_vertex = function("backdrop_pass_vertex")?;
        let rect_vertex = function("backdrop_blur_rect_vertex")?;
        Ok(Self {
            format,
            downsample: pipeline(
                device,
                "backdrop_blur_downsample",
                &pass_vertex,
                &*function("backdrop_blur_downsample_fragment")?,
                format,
            )?,
            upsample: pipeline(
                device,
                "backdrop_blur_upsample",
                &pass_vertex,
                &*function("backdrop_blur_upsample_fragment")?,
                format,
            )?,
            frame_copy: pipeline(
                device,
                "backdrop_frame_copy",
                &pass_vertex,
                &*function("backdrop_frame_copy_fragment")?,
                format,
            )?,
            composite: pipeline(
                device,
                "backdrop_blur_rect",
                &rect_vertex,
                &*function("backdrop_blur_rect_fragment")?,
                format,
            )?,
        })
    }
}

/// A pipeline that rewrites every pixel it covers: blending off.
fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    vertex: &ProtocolObject<dyn MTLFunction>,
    fragment: &ProtocolObject<dyn MTLFunction>,
    format: MTLPixelFormat,
) -> Result<Pipeline> {
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex));
    descriptor.setFragmentFunction(Some(fragment));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setPixelFormat(format);
    attachment.setBlendingEnabled(false);
    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn texture(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    width: usize,
    height: usize,
    format: MTLPixelFormat,
    usage: MTLTextureUsage,
) -> Result<Texture> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: Both dimensions are at least one and this descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(width.max(1));
        descriptor.setHeight(height.max(1));
    }
    descriptor.setPixelFormat(format);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(usage);
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .with_context(|| format!("allocating {label}"))?;
    texture.setLabel(Some(&NSString::from_str(label)));
    Ok(texture)
}

/// The deepest pyramid a scene's blurs need, counting the blurs inside its rounded clip groups (whose scenes the
/// groups carry); `None` when it has none. Layer tiles never hold one (`gpui::fast::layers::paint`).
fn levels_needed(scene: &Scene) -> Option<usize> {
    let own = scene
        .backdrop_blur_rects
        .iter()
        .map(|rect| rect.effective_kernel_levels() as usize)
        .max();
    let nested = scene
        .layers
        .frames
        .iter()
        .filter(|frame| frame.tile_size == 0)
        .filter_map(|frame| frame.content.scene())
        .filter_map(levels_needed)
        .max();
    own.max(nested).map(|levels| levels.min(LEVELS))
}

#[cfg(test)]
thread_local! {
    /// Renders through the intermediate a framebuffer-only drawable gets, into a target that can still be read back.
    pub(crate) static FORCE_INTERMEDIATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Blurs the whole frame, the reference the region-limited passes must reproduce.
    pub(crate) static FULL_FRAME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Called at the start of a frame, before anything is encoded: decides where it renders. A frame without a blur
/// renders into `drawable` and allocates nothing.
pub(crate) fn prepare_frame(
    renderer: &mut MetalRenderer,
    scene: &Scene,
    drawable: &ProtocolObject<dyn MTLTexture>,
) -> FrameTarget {
    let direct = FrameTarget {
        texture: drawable.retain(),
        copy_to: None,
    };
    let needed = levels_needed(scene);
    let backdrop = &mut renderer.backdrop;
    backdrop.frame_enabled = false;
    backdrop.trim(needed.map_or(0, |levels| levels + 1));
    if backdrop.targets.is_none() && backdrop.framebuffer_only_lifted {
        backdrop.framebuffer_only_lifted = false;
        if let Some(layer) = &renderer.layer {
            layer.setFramebufferOnly(true);
        }
    }
    if needed.is_none() {
        return direct;
    }
    let format = drawable.pixelFormat();
    if !backdrop.ensure_pipelines(&renderer.device, format) {
        return direct;
    }
    let framebuffer_only = drawable.isFramebufferOnly();
    #[cfg(test)]
    let framebuffer_only = framebuffer_only || FORCE_INTERMEDIATE.get();
    if !framebuffer_only {
        backdrop.frame_enabled = true;
        return direct;
    }
    // The drawables vended from now on can be copied from; this one cannot.
    if let Some(layer) = &renderer.layer
        && layer.framebufferOnly()
    {
        layer.setFramebufferOnly(false);
        backdrop.framebuffer_only_lifted = true;
    }
    let (width, height) = (drawable.width(), drawable.height());
    let intermediate = backdrop
        .ensure_targets(&renderer.device, width, height, format, 0)
        .and_then(|targets| {
            if let Some(frame) = &targets.frame {
                return Ok(frame.clone());
            }
            let frame = texture(
                &renderer.device,
                "backdrop_frame",
                width,
                height,
                format,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            )?;
            targets.frame = Some(frame.clone());
            Ok(frame)
        });
    match intermediate {
        Ok(intermediate) => {
            backdrop.frame_enabled = true;
            FrameTarget {
                texture: intermediate,
                copy_to: Some(drawable.retain()),
            }
        }
        Err(error) => {
            log::warn!(
                "a frame's backdrop blurs are skipped: its intermediate could not be made: {error:#}"
            );
            direct
        }
    }
}

/// Copies an intermediate frame onto the drawable it stood in for; nothing for a direct frame.
pub(crate) fn finish_frame(
    renderer: &MetalRenderer,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    frame: &FrameTarget,
) -> Result<()> {
    let Some(drawable) = &frame.copy_to else {
        return Ok(());
    };
    let pipelines = renderer
        .backdrop
        .pipelines
        .as_ref()
        .context("backdrop pipelines missing")?;
    // Every pixel is overwritten with blending off.
    let encoder = render_pass(command_buffer, drawable, MTLLoadAction::DontCare)?;
    encoder.setRenderPipelineState(&pipelines.frame_copy);
    // SAFETY: The retained intermediate matches the shader's texture slot, and the strip is the four vertices the
    // vertex stage derives.
    unsafe {
        encoder.setFragmentTexture_atIndex(Some(&frame.texture), PROCESSED_TEXTURE);
        encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
    }
    encoder.endEncoding();
    Ok(())
}

/// Draws a batch of blurs (`rects` is the batch, at `start` in the scene's instances): ends `encoder`, and for each
/// group of rects that share a kernel depth and do not overlap, snapshots and blurs the frame and composites them;
/// returns the encoder the frame continues in, on the same target (`rebind` restores what the caller had bound on
/// it, a rounded clip group's stack). A frame whose blurs are disabled (no pipelines, an unreadable target) returns
/// `encoder` as it was: the blurs are skipped, tint included.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw(
    backdrop: &mut Backdrop,
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    mut encoder: Encoder,
    target: &ProtocolObject<dyn MTLTexture>,
    rects: &[BackdropBlurRect],
    start: usize,
    instances: &InstanceBinding,
    viewport_size: Size<DevicePixels>,
    rebind: impl Fn(&ProtocolObject<dyn MTLRenderCommandEncoder>),
) -> Result<Encoder> {
    if rects.is_empty() || !backdrop.frame_enabled || target.isFramebufferOnly() {
        return Ok(encoder);
    }
    let frame_rect = PixelRect::of(target);
    for group in groups(rects) {
        let levels = (rects[group.start].effective_kernel_levels() as usize).min(LEVELS);
        let Some(union) = rects[group.clone()]
            .iter()
            .map(|rect| rect.bounds.intersect(&rect.content_mask.bounds))
            .reduce(|union, bounds| union.union(&bounds))
        else {
            continue;
        };
        if PixelRect::round_out(&union, 0)
            .intersect(frame_rect)
            .is_empty()
        {
            continue;
        }
        // Only the pixels the composite reads, plus the kernel's reach.
        let region = PixelRect::round_out(&union, reach(levels));
        #[cfg(test)]
        let region = if FULL_FRAME.get() { frame_rect } else { region };
        encoder.endEncoding();
        let (processed, original) =
            snapshot(backdrop, device, command_buffer, target, levels, region)?;
        let pipelines = backdrop
            .pipelines
            .as_ref()
            .context("backdrop pipelines missing")?;
        encoder = new_command_encoder_for_texture(command_buffer, target, viewport_size, None)?;
        rebind(&encoder);
        encoder.setRenderPipelineState(&pipelines.composite);
        let range = start + group.start..start + group.end;
        // SAFETY: The binding holds the scene's blur rects at an aligned offset and `range` lies within them; Metal
        // retains the buffer and the textures, and copies the viewport bytes from a live value into the slots the
        // shaders declare. The strip is the four vertices the vertex stage derives.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&instances.buffer),
                instances.offset,
                INSTANCES_INDEX,
            );
            encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                VIEWPORT_INDEX,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&instances.buffer),
                instances.offset,
                INSTANCES_INDEX,
            );
            encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                VIEWPORT_INDEX,
            );
            encoder.setFragmentTexture_atIndex(Some(&processed), PROCESSED_TEXTURE);
            encoder.setFragmentTexture_atIndex(Some(&original), ORIGINAL_TEXTURE);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                range.len(),
                range.start,
            );
        }
    }
    Ok(encoder)
}

/// Copies `region` of what `target` holds into the snapshot and blurs it `levels` deep; returns (the processed
/// copy, the untouched one). Called between encoders: it opens and ends its own.
fn snapshot(
    backdrop: &mut Backdrop,
    device: &ProtocolObject<dyn MTLDevice>,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    target: &ProtocolObject<dyn MTLTexture>,
    levels: usize,
    region: PixelRect,
) -> Result<(Texture, Texture)> {
    let format = target.pixelFormat();
    anyhow::ensure!(
        backdrop.ensure_pipelines(device, format),
        "backdrop pipelines missing for the target's format"
    );
    backdrop.ensure_targets(device, target.width(), target.height(), format, levels)?;
    let backdrop = &*backdrop;
    let pipelines = backdrop
        .pipelines
        .as_ref()
        .context("backdrop pipelines missing")?;
    let targets = backdrop
        .targets
        .as_ref()
        .context("backdrop targets missing")?;
    let region = region.intersect(PixelRect::of(target));
    anyhow::ensure!(!region.is_empty(), "empty backdrop region");

    let blit = command_buffer
        .blitCommandEncoder()
        .context("creating the backdrop's blit encoder")?;
    let origin = MTLOrigin {
        x: region.left as usize,
        y: region.top as usize,
        z: 0,
    };
    // SAFETY: Both textures are 2D, of the target's size and format, and the region lies within them.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            target,
            0,
            0,
            origin,
            MTLSize {
                width: (region.right - region.left) as usize,
                height: (region.bottom - region.top) as usize,
                depth: 1,
            },
            &targets.snapshot,
            0,
            0,
            origin,
        );
    }
    blit.endEncoding();

    // A pass writes only its scissor and leaves the rest of the level as it was; the rects are rounded out so every
    // texel the next pass reads lies inside what this one wrote.
    let pass = |pipeline: &Pipeline,
                source: &Texture,
                destination: &Texture,
                scale: usize|
     -> Result<()> {
        let rect = region
            .scale_down(scale)
            .intersect(PixelRect::of(destination));
        if rect.is_empty() {
            return Ok(());
        }
        let encoder = render_pass(command_buffer, destination, MTLLoadAction::Load)?;
        encoder.setRenderPipelineState(pipeline);
        encoder.setScissorRect(MTLScissorRect {
            x: rect.left as usize,
            y: rect.top as usize,
            width: (rect.right - rect.left) as usize,
            height: (rect.bottom - rect.top) as usize,
        });
        // SAFETY: The retained source matches the shader's texture slot, and the strip is the four vertices the vertex
        // stage derives.
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(source), PROCESSED_TEXTURE);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
        }
        encoder.endEncoding();
        Ok(())
    };

    let processed = if levels == 0 {
        &targets.snapshot
    } else {
        let pyramid = targets
            .levels
            .get(..levels)
            .context("missing blur pyramid levels")?;
        pass(&pipelines.downsample, &targets.snapshot, &pyramid[0], 1)?;
        for level in 1..levels {
            pass(
                &pipelines.downsample,
                &pyramid[level - 1],
                &pyramid[level],
                level + 1,
            )?;
        }
        for level in (1..levels).rev() {
            pass(
                &pipelines.upsample,
                &pyramid[level],
                &pyramid[level - 1],
                level,
            )?;
        }
        &pyramid[0]
    };
    Ok((processed.clone(), targets.snapshot.clone()))
}

/// A render pass over all of `texture`, stored.
fn render_pass(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    load: MTLLoadAction,
) -> Result<Encoder> {
    let descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setTexture(Some(texture));
    attachment.setLoadAction(load);
    attachment.setStoreAction(MTLStoreAction::Store);
    command_buffer
        .renderCommandEncoderWithDescriptor(&descriptor)
        .context("creating a backdrop render command encoder")
}

/// Whole device pixels, right and bottom edges exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PixelRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl PixelRect {
    fn of(texture: &ProtocolObject<dyn MTLTexture>) -> Self {
        Self {
            left: 0,
            top: 0,
            right: texture.width() as i32,
            bottom: texture.height() as i32,
        }
    }

    /// The pixels covering `bounds`, grown by `margin` on every side.
    fn round_out(bounds: &Bounds<ScaledPixels>, margin: i32) -> Self {
        let clamp = |value: f32| value.clamp(i32::MIN as f32 / 2., i32::MAX as f32 / 2.) as i32;
        Self {
            left: clamp(bounds.origin.x.0.floor()).saturating_sub(margin),
            top: clamp(bounds.origin.y.0.floor()).saturating_sub(margin),
            right: clamp((bounds.origin.x.0 + bounds.size.width.0).ceil()).saturating_add(margin),
            bottom: clamp((bounds.origin.y.0 + bounds.size.height.0).ceil()).saturating_add(margin),
        }
    }

    fn intersect(self, other: Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }

    fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    /// This rect in a level `2^level` times smaller, rounded out by a texel.
    fn scale_down(self, level: usize) -> Self {
        let divisor = 1i32 << level;
        Self {
            left: self.left.div_euclid(divisor) - 1,
            top: self.top.div_euclid(divisor) - 1,
            right: (self.right + divisor - 1).div_euclid(divisor) + 1,
            bottom: (self.bottom + divisor - 1).div_euclid(divisor) + 1,
        }
    }
}

/// Frame pixels a blur of `levels` pyramid levels can carry into a pixel: each pass reads a texel to either side of its
/// own, `2 * 2^level` frame pixels at level `level`, both chains together stay below `4 * 2^levels`, and the
/// composite's bilinear read adds a few more.
fn reach(levels: usize) -> i32 {
    (4 << levels) + 8
}

/// Splits a batch into runs that share a kernel depth and do not overlap.
fn groups(rects: &[BackdropBlurRect]) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut group_start = 0;
    while group_start < rects.len() {
        let levels = rects[group_start].effective_kernel_levels();
        let mut group_end = group_start + 1;
        while group_end < rects.len() {
            let candidate = &rects[group_end];
            if candidate.effective_kernel_levels() != levels
                || rects[group_start..group_end]
                    .iter()
                    .any(|previous| previous.bounds.intersects(&candidate.bounds))
            {
                break;
            }
            group_end += 1;
        }
        groups.push(group_start..group_end);
        group_start = group_end;
    }
    groups
}

#[cfg(test)]
#[path = "backdrop_tests.rs"]
mod tests;
