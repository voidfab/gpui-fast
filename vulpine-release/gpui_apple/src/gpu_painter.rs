// Modified by Longbridge for gpui-fast.
//! vulpine (FOX9-f0a295): the in-frame GPU painter's Metal backend.
//!
//! Ported from Pomelo's GPUI fork (HaiwenZhang/zed, branch `gpui-pre-native-gpu` at 0edb2919,
//! `crates/gpui_apple/src/gpu_painter.rs` and its `metal_renderer.rs` hooks, Apache-2.0) onto gpui-fast-apple 0.1.9.
//! A painter gets GPUI's own `MTLDevice` and the frame's command buffer. It draws in one of two ways:
//!
//! - **Shared pass** ([`MetalSharedPainter`], [`metal_shared_painter`]): inside GPUI's current render pass, at the
//!   paint's place in the draw order, with the viewport set to the element's bounds and the scissor to its clip.
//!   Inside a rounded clip group the group's clip stack is bound where GPUI's own fragment shaders read it (fragment
//!   buffers 28, the clips, and 29, their count) and handed over in [`MetalDrawContext::rounded_clips`], so a fragment
//!   shader can keep the rounded corners.
//! - **Own pass** ([`MetalPainter`], [`metal_painter`]): GPUI's pass is ended, the painter encodes passes of its own on
//!   the frame's command buffer (Load/Store to keep GPUI's colour), and GPUI's pass begins again after it.
//!
//! Every visible paint is prepared first ([`MetalPainter::prepare`]), on a command buffer of its own that is
//! committed before the frame's, so offscreen work (a 3D scene with depth into the painter's own texture) runs before
//! GPUI draws. gpui-fast's Metal renderer keeps one device for its life: [`GpuResetReason::DeviceReplaced`] is never
//! sent here; [`GpuResetReason::WindowDestroyed`] comes from the window's registry when the window goes.

use std::collections::HashSet;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use block2::RcBlock;
use gpui::{
    DevicePixels, GpuDrawMode, GpuPaintSurface, GpuPaintTarget, GpuPainter, GpuPainterBackend,
    GpuResetReason, Scene, Size,
};
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::any::Any;
use std::sync::Mutex;

use crate::metal_renderer::{MetalRenderer, new_command_encoder_for_texture};

/// Native Metal resources borrowed before scene drawing begins.
///
/// Use application-owned targets. Do not submit or present, retain frame resources, reenter GPUI, or leave an
/// encoder active on return.
pub struct MetalPrepareContext<'a> {
    /// Geometry and opacity of the surface that will be drawn later.
    pub target: GpuPaintTarget,
    /// Resources must be created on this device.
    pub device: &'a metal::DeviceRef,
    /// Commands recorded here execute before the frame's.
    pub command_buffer: &'a metal::CommandBufferRef,
    /// Colour format of the eventual drawing target.
    pub color_format: metal::MTLPixelFormat,
}

/// Native Metal drawing access within GPUI's current pass. GPUI sets the viewport (the element's bounds) and the
/// scissor (its clip) first. Do not end the pass, retain its resources, or reenter GPUI.
pub struct MetalDrawContext<'a> {
    /// Target geometry and inherited opacity; apply opacity when writing colour.
    pub target: GpuPaintTarget,
    /// Resources must be created on this device.
    pub device: &'a metal::DeviceRef,
    /// The current UI pass, borrowed only for this callback.
    pub encoder: &'a metal::RenderCommandEncoderRef,
    /// Colour format of the target.
    pub color_format: metal::MTLPixelFormat,
    /// The enclosing rounded clip groups, innermost last: each `[left, top, right, bottom, four horizontal radii,
    /// four vertical radii]` in device pixels (radii top-left, top-right, bottom-right, bottom-left). Also bound at
    /// fragment buffer 28 with its count at 29, as GPUI's shaders read them.
    pub rounded_clips: &'a [[f32; 12]],
}

/// Native Metal resources borrowed for one own-pass drawing callback.
pub struct MetalPaintContext<'a> {
    /// Target geometry and inherited opacity.
    pub target: GpuPaintTarget,
    /// Resources must be created on this device.
    pub device: &'a metal::DeviceRef,
    /// Begin and end application encoders here; never commit it.
    pub command_buffer: &'a metal::CommandBufferRef,
    /// Use Load/Store to keep GPUI's colour.
    pub color_target: &'a metal::TextureRef,
    /// Colour format of the target.
    pub color_format: metal::MTLPixelFormat,
    /// The enclosing rounded clip groups, as [`MetalDrawContext::rounded_clips`].
    pub rounded_clips: &'a [[f32; 12]],
}

/// Own-pass drawing with the window's native Metal resources.
pub trait MetalPainter: Send + 'static {
    /// Immutable application data retained by the scene, including scene replays.
    type Snapshot: Send + Sync + 'static;

    /// Prepares resources before the frame, once per visible paint. An error skips this paint's drawing.
    fn prepare(
        &mut self,
        _context: &mut MetalPrepareContext<'_>,
        _snapshot: &Self::Snapshot,
    ) -> Result<()> {
        Ok(())
    }

    /// Draws at this paint's place in the scene. Keep existing colour, obey the clip, apply the opacity. Do not
    /// submit, present, retain frame resources or reenter GPUI; end any encoder before returning, error or not.
    fn paint(&mut self, context: &mut MetalPaintContext<'_>, snapshot: &Self::Snapshot) -> Result<()>;

    /// Releases resources before the window goes.
    fn reset(&mut self, reason: GpuResetReason);
}

/// Drawing in GPUI's existing Metal pass. Attachments cannot be replaced; leave the pass open.
pub trait MetalSharedPainter: Send + 'static {
    /// Immutable application data retained by the scene, including scene replays.
    type Snapshot: Send + Sync + 'static;

    /// Prepares resources before the frame, once per visible paint. An error skips this paint's drawing.
    fn prepare(
        &mut self,
        _context: &mut MetalPrepareContext<'_>,
        _snapshot: &Self::Snapshot,
    ) -> Result<()> {
        Ok(())
    }

    /// Draws at this paint's place in the scene. Keep existing colour, apply the opacity.
    fn paint(&mut self, context: &mut MetalDrawContext<'_>, snapshot: &Self::Snapshot) -> Result<()>;

    /// Releases resources before the window goes.
    fn reset(&mut self, reason: GpuResetReason);
}

/// Registers own-pass drawing with typed application snapshots.
pub fn metal_painter<P: MetalPainter>(painter: P) -> GpuPainter<P::Snapshot> {
    GpuPainter::new(MetalPainterAdapter(
        Box::new(Independent(painter)),
        GpuDrawMode::OwnPass,
    ))
}

/// Registers shared-pass drawing with typed application snapshots.
pub fn metal_shared_painter<P: MetalSharedPainter>(painter: P) -> GpuPainter<P::Snapshot> {
    GpuPainter::new(MetalPainterAdapter(
        Box::new(Shared(painter)),
        GpuDrawMode::SharedPass,
    ))
}

pub(crate) struct MetalPainterAdapter(Box<dyn AnyMetalPainter>, GpuDrawMode);

impl GpuPainterBackend for MetalPainterAdapter {
    fn draw_mode(&self) -> GpuDrawMode {
        self.1
    }

    fn reset(&mut self, reason: GpuResetReason) {
        self.0.reset(reason);
    }
}

type AnySnapshot = dyn Any + Send + Sync;

trait AnyMetalPainter: Send {
    fn prepare(&mut self, context: &mut MetalPrepareContext<'_>, snapshot: &AnySnapshot) -> Result<()>;
    fn paint(&mut self, _: &mut MetalPaintContext<'_>, _: &AnySnapshot) -> Result<()> {
        anyhow::bail!("own-pass access asked of a shared-pass painter")
    }
    fn draw_in_pass(&mut self, _: &mut MetalDrawContext<'_>, _: &AnySnapshot) -> Result<()> {
        anyhow::bail!("shared-pass access asked of an own-pass painter")
    }
    fn reset(&mut self, reason: GpuResetReason);
}

fn snapshot<T: 'static>(snapshot: &AnySnapshot) -> Result<&T> {
    snapshot
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow::anyhow!("GPU painter snapshot type mismatch"))
}

struct Independent<P>(P);

impl<P: MetalPainter> AnyMetalPainter for Independent<P> {
    fn prepare(&mut self, context: &mut MetalPrepareContext<'_>, data: &AnySnapshot) -> Result<()> {
        self.0.prepare(context, snapshot(data)?)
    }
    fn paint(&mut self, context: &mut MetalPaintContext<'_>, data: &AnySnapshot) -> Result<()> {
        self.0.paint(context, snapshot(data)?)
    }
    fn reset(&mut self, reason: GpuResetReason) {
        self.0.reset(reason);
    }
}

struct Shared<P>(P);

impl<P: MetalSharedPainter> AnyMetalPainter for Shared<P> {
    fn prepare(&mut self, context: &mut MetalPrepareContext<'_>, data: &AnySnapshot) -> Result<()> {
        self.0.prepare(context, snapshot(data)?)
    }
    fn draw_in_pass(&mut self, context: &mut MetalDrawContext<'_>, data: &AnySnapshot) -> Result<()> {
        self.0.paint(context, snapshot(data)?)
    }
    fn reset(&mut self, reason: GpuResetReason) {
        self.0.reset(reason);
    }
}

/// What the renderer keeps for painters between a frame's preparation and its drawing.
#[derive(Default)]
pub(crate) struct RendererState {
    /// Paints (by address in the frame's scenes) whose preparation failed this frame: not drawn.
    unready: HashSet<usize>,
    /// GPUI's pass state, restored after a shared-pass painter.
    depth_stencil: Option<metal::DepthStencilState>,
}

fn key(surface: &GpuPaintSurface) -> usize {
    surface as *const GpuPaintSurface as usize
}

fn size_of(texture: &metal::TextureRef) -> [u32; 2] {
    [texture.width() as u32, texture.height() as u32]
}

/// Every paint of `scene` and of the rounded clip groups inside it, in drawing order per scene.
fn each_paint<'s>(scene: &'s Scene, out: &mut Vec<&'s GpuPaintSurface>) {
    out.extend(scene.gpu_paints.iter());
    for frame in &scene.layers.frames {
        if frame.tile_size == 0
            && let Some(child) = frame.content.scene()
        {
            each_paint(child, out);
        }
    }
}

/// Forwarded to by `MetalRenderer::render_frame` before it encodes `scene`: prepares every visible paint on a
/// command buffer committed ahead of the frame's, and notes the ones whose preparation failed.
pub(crate) fn prepare_frame(renderer: &mut MetalRenderer, scene: &Scene, texture: &metal::TextureRef) {
    FRAMES.fetch_add(1, Ordering::Relaxed);
    renderer.gpu_paint.unready.clear();
    let mut paints = Vec::new();
    each_paint(scene, &mut paints);
    if paints.is_empty() {
        return;
    }
    let size = size_of(texture);
    let samples = texture.sample_count() as u32;
    let queue = renderer.command_queue.clone();
    let command_buffer = queue.new_command_buffer();
    command_buffer.set_label("gpu_paint_prepare");
    let mut unready = Vec::new();
    for surface in paints {
        let draw = &surface.draw;
        let target = draw.target(size, surface.bounds, surface.content_mask, samples);
        let [_, _, width, height] = target.scissor_rect();
        if width == 0 || height == 0 {
            continue;
        }
        let prepared = draw.handle.prepare::<MetalPainterAdapter>(|painter| {
            painter.0.prepare(
                &mut MetalPrepareContext {
                    target,
                    device: &renderer.device,
                    command_buffer,
                    color_format: texture.pixel_format(),
                },
                draw.data.as_ref(),
            )
        });
        if let Err(error) = prepared {
            log::error!("GPU painter preparation failed: {error:#}");
            unready.push(key(surface));
        }
    }
    command_buffer.commit();
    renderer.gpu_paint.unready.extend(unready);
}

/// Draws a batch of paints at its place in the frame: `encoder` is GPUI's current pass on `command_buffer`, and is
/// replaced when an own-pass painter ends it. `rounded_clips` are the enclosing rounded clip groups (bound already).
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw<'a>(
    renderer: &mut MetalRenderer,
    surfaces: &[GpuPaintSurface],
    texture: &'a metal::TextureRef,
    command_buffer: &'a metal::CommandBufferRef,
    encoder: &mut &'a metal::RenderCommandEncoderRef,
    viewport: Size<DevicePixels>,
    rounded_clips: &[[f32; 12]],
) {
    let size = size_of(texture);
    let samples = texture.sample_count() as u32;
    for surface in surfaces {
        if renderer.gpu_paint.unready.contains(&key(surface)) {
            continue;
        }
        let draw = &surface.draw;
        let target = draw.target(size, surface.bounds, surface.content_mask, samples);
        let [left, top, width, height] = target.scissor_rect();
        if width == 0 || height == 0 {
            continue;
        }
        let mode = draw.handle.draw_mode();
        let device = renderer.device.clone();
        let result = draw.handle.invoke::<MetalPainterAdapter>(|painter| match mode {
            GpuDrawMode::SharedPass => {
                encoder.set_viewport(metal::MTLViewport {
                    originX: target.bounds.origin.x.0 as f64,
                    originY: target.bounds.origin.y.0 as f64,
                    width: target.bounds.size.width.0 as f64,
                    height: target.bounds.size.height.0 as f64,
                    znear: 0.0,
                    zfar: 1.0,
                });
                encoder.set_scissor_rect(metal::MTLScissorRect {
                    x: left as u64,
                    y: top as u64,
                    width: width as u64,
                    height: height as u64,
                });
                painter.0.draw_in_pass(
                    &mut MetalDrawContext {
                        target,
                        device: &device,
                        encoder: *encoder,
                        color_format: texture.pixel_format(),
                        rounded_clips,
                    },
                    draw.data.as_ref(),
                )
            }
            GpuDrawMode::OwnPass => {
                encoder.end_encoding();
                let result = painter.0.paint(
                    &mut MetalPaintContext {
                        target,
                        device: &device,
                        command_buffer,
                        color_target: texture,
                        color_format: texture.pixel_format(),
                        rounded_clips,
                    },
                    draw.data.as_ref(),
                );
                *encoder = new_command_encoder_for_texture(command_buffer, texture, viewport, None);
                result
            }
        });
        if mode == GpuDrawMode::SharedPass {
            restore_ui_state(renderer, *encoder, size, viewport);
        }
        if let Err(error) = result {
            log::error!("GPU painter draw failed: {error:#}");
        }
    }
}

/// Puts GPUI's pass back as `new_command_encoder_for_texture` leaves it, after a shared-pass painter.
fn restore_ui_state(
    renderer: &mut MetalRenderer,
    encoder: &metal::RenderCommandEncoderRef,
    size: [u32; 2],
    viewport: Size<DevicePixels>,
) {
    encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: viewport.width.0 as f64,
        height: viewport.height.0 as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(metal::MTLScissorRect {
        x: 0,
        y: 0,
        width: size[0] as u64,
        height: size[1] as u64,
    });
    encoder.set_cull_mode(metal::MTLCullMode::None);
    encoder.set_front_facing_winding(metal::MTLWinding::Clockwise);
    encoder.set_triangle_fill_mode(metal::MTLTriangleFillMode::Fill);
    encoder.set_depth_clip_mode(metal::MTLDepthClipMode::Clip);
    encoder.set_depth_bias(0.0, 0.0, 0.0);
    let device = renderer.device.clone();
    let depth_stencil = renderer
        .gpu_paint
        .depth_stencil
        .get_or_insert_with(|| device.new_depth_stencil_state(&metal::DepthStencilDescriptor::new()));
    encoder.set_depth_stencil_state(depth_stencil);
    encoder.set_stencil_reference_value(0);
    encoder.set_blend_color(0.0, 0.0, 0.0, 0.0);
    encoder.set_visibility_result_mode(metal::MTLVisibilityResultMode::Disabled, 0);
    crate::fast::rounded_clip::default_mask(encoder);
}

/// Frames the Metal renderers of this process have encoded (every `render_frame`), for at-rest counts.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// How many frames this process's GPUI renderers have encoded so far.
pub fn frames_encoded() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Frame GPU times in ms, kept only while `VULPINE_GPU_PAINT_TIMING` is set (measurement runs).
static GPU_MS: Mutex<Vec<f64>> = Mutex::new(Vec::new());

fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VULPINE_GPU_PAINT_TIMING").is_some())
}

/// With `VULPINE_GPU_PAINT_TIMING` set, records a frame's GPU time when its command buffer completes.
pub(crate) fn time_frame(commands: &metal::CommandBufferRef) {
    if !timing() {
        return;
    }
    let block = RcBlock::new(|buffer: ptr::NonNull<AnyObject>| {
        // SAFETY: Metal hands its completion handlers the command buffer that completed; both reads are
        // CFTimeInterval properties of it.
        let buffer = unsafe { buffer.as_ref() };
        let start: f64 = unsafe { msg_send![buffer, GPUStartTime] };
        let end: f64 = unsafe { msg_send![buffer, GPUEndTime] };
        if let Ok(mut times) = GPU_MS.lock() {
            times.push((end - start) * 1000.0);
        }
    });
    // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI, as `render_frame`
    // relies on too.
    unsafe {
        commands.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
    }
}

/// The frame GPU times recorded since the last call (empty unless `VULPINE_GPU_PAINT_TIMING` is set).
pub fn take_frame_gpu_ms() -> Vec<f64> {
    GPU_MS
        .lock()
        .map(|mut times| std::mem::take(&mut *times))
        .unwrap_or_default()
}
