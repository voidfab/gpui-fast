//! Application GPU drawing ordered with ordinary GPUI primitives (vulpine's in-frame painter, FOX9-f0a295).
//!
//! Ported from Pomelo's GPUI fork (HaiwenZhang/zed, branch `gpui-pre-native-gpu` at 0edb2919,
//! `crates/gpui/src/gpu_painter.rs`, Apache-2.0) onto gpui-fast 0.1.9 and cut down: one preparation stage, the
//! registry held by the window rather than the platform renderer. A painter is registered once per window
//! ([`crate::Window::register_gpu_painter`]) and drawn with a snapshot inside a paint closure
//! ([`crate::Window::paint_gpu`]). The scene keeps the draw as a primitive of its own, so it takes the current draw
//! order, content mask and opacity like any quad: it is clipped by its ancestors, replayed with cached views, moved
//! into rounded clip groups, and drawn under whatever is painted after it. The native backend (gpui-fast-apple's
//! `gpu_painter`) hands the painter the window's own device and the frame's command buffer.

use crate::{AnyWindowHandle, Bounds, ContentMask, ScaledPixels};
use anyhow::Result;
use parking_lot::Mutex;
use std::{
    any::Any,
    fmt,
    marker::PhantomData,
    sync::{Arc, Weak},
};

/// Backend adapter used to register native drawing callbacks.
#[doc(hidden)]
pub trait GpuPainterBackend: Any + Send {
    /// Drawing access fixed by the native registration adapter.
    fn draw_mode(&self) -> GpuDrawMode {
        GpuDrawMode::OwnPass
    }

    /// Releases resources before the window's device or renderer is retired. Resize alone does not call it.
    fn reset(&mut self, reason: GpuResetReason);
}

/// A native painter awaiting registration, accepting snapshots of type `Snapshot`.
///
/// Construct with the painter adapters the native backend crates export (`gpui_apple::metal_shared_painter`).
pub struct GpuPainter<Snapshot> {
    painter: Box<dyn GpuPainterBackend>,
    snapshot: PhantomData<fn(Snapshot) -> Snapshot>,
}

impl<Snapshot: Send + Sync + 'static> GpuPainter<Snapshot> {
    #[doc(hidden)]
    pub fn new(painter: impl GpuPainterBackend) -> Self {
        Self {
            painter: Box::new(painter),
            snapshot: PhantomData,
        }
    }

    pub(crate) fn register(
        self,
        owner: AnyWindowHandle,
    ) -> (GpuPainterHandle<Snapshot>, GpuPainterRegistration) {
        let (handle, registration) = AnyGpuPainterHandle::from_box(owner, self.painter);
        (
            GpuPainterHandle {
                handle,
                snapshot: PhantomData,
            },
            registration,
        )
    }
}

/// A window-scoped native painter accepting snapshots of type `Snapshot`.
pub struct GpuPainterHandle<Snapshot> {
    pub(crate) handle: AnyGpuPainterHandle,
    snapshot: PhantomData<fn(Snapshot) -> Snapshot>,
}

impl<Snapshot> Clone for GpuPainterHandle<Snapshot> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            snapshot: PhantomData,
        }
    }
}

impl<Snapshot> fmt::Debug for GpuPainterHandle<Snapshot> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.handle.fmt(formatter)
    }
}

impl<Snapshot> GpuPainterHandle<Snapshot> {
    /// Encoding diagnostics. Do not call from this painter's callbacks (they hold the same lock).
    pub fn status(&self) -> GpuPainterStatus {
        self.handle.status()
    }

    /// Takes the latest callback error. Successful draws do not clear errors.
    pub fn take_error(&self) -> Option<String> {
        self.handle.take_error()
    }
}

/// The drawing access requested by an application GPU painter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuDrawMode {
    /// Draw into GPUI's current pass without replacing its attachments.
    SharedPass,
    /// End GPUI's pass and let the painter encode passes of its own on the frame's command buffer.
    #[default]
    OwnPass,
}

/// Why a painter must release its resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuResetReason {
    /// The owning window's device is being replaced.
    DeviceReplaced,
    /// The owning window is being destroyed.
    WindowDestroyed,
}

/// Geometry in physical pixels in the current target's coordinate system.
#[derive(Clone, Copy, Debug)]
pub struct GpuPaintTarget {
    /// Full color target dimensions.
    pub size: [u32; 2],
    /// Complete element bounds, even when partially clipped.
    pub bounds: Bounds<ScaledPixels>,
    /// Intersection of bounds, parent clip and target edges.
    pub clip: Bounds<ScaledPixels>,
    /// Logical to physical pixel scale.
    pub scale_factor: f32,
    /// Samples per color pixel; application attachments must match.
    pub sample_count: u32,
    /// Inherited element opacity; apply when writing color.
    pub opacity: f32,
}

impl GpuPaintTarget {
    /// An integer scissor `[left, top, width, height]` inside the clip and the target. Fractional edges round
    /// inward so writes cannot escape the inherited clip; a subpixel clip may give a zero width or height.
    pub fn scissor_rect(&self) -> [u32; 4] {
        let left = self.clip.origin.x.0.ceil().clamp(0.0, self.size[0] as f32) as u32;
        let top = self.clip.origin.y.0.ceil().clamp(0.0, self.size[1] as f32) as u32;
        let right = self.clip.right().0.floor().clamp(0.0, self.size[0] as f32) as u32;
        let bottom = self.clip.bottom().0.floor().clamp(0.0, self.size[1] as f32) as u32;
        [
            left,
            top,
            right.saturating_sub(left),
            bottom.saturating_sub(top),
        ]
    }
}

/// Encoding diagnostics; these do not establish GPU execution completion.
#[derive(Clone, Debug, Default)]
pub struct GpuPainterStatus {
    /// Successful drawing callback encodings.
    pub encoded: u64,
    /// Lifecycle notifications.
    pub resets: u64,
    /// Most recent preparation or drawing error, kept until taken.
    pub last_error: Option<String>,
}

struct PainterState {
    painter: Box<dyn GpuPainterBackend>,
    status: GpuPainterStatus,
}

/// Type-erased registration retained by referencing scenes.
#[doc(hidden)]
#[derive(Clone)]
pub struct AnyGpuPainterHandle {
    owner: AnyWindowHandle,
    state: Arc<Mutex<PainterState>>,
}

impl fmt::Debug for AnyGpuPainterHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuPainterHandle")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl AnyGpuPainterHandle {
    fn from_box(
        owner: AnyWindowHandle,
        painter: Box<dyn GpuPainterBackend>,
    ) -> (Self, GpuPainterRegistration) {
        let state = Arc::new(Mutex::new(PainterState {
            painter,
            status: GpuPainterStatus::default(),
        }));
        let registration = GpuPainterRegistration(Arc::downgrade(&state));
        (Self { owner, state }, registration)
    }

    pub(crate) fn owner(&self) -> AnyWindowHandle {
        self.owner
    }

    /// The painter's encoding diagnostics.
    pub fn status(&self) -> GpuPainterStatus {
        self.state.lock().status.clone()
    }

    /// Takes the latest callback error.
    pub fn take_error(&self) -> Option<String> {
        self.state.lock().status.last_error.take()
    }

    /// Runs a drawing callback on the painter as the backend's adapter type `T`, counting a success as one
    /// encoding and keeping an error in the status.
    #[doc(hidden)]
    pub fn invoke<T: GpuPainterBackend>(
        &self,
        paint: impl FnOnce(&mut T) -> Result<()>,
    ) -> Result<()> {
        self.invoke_callback(paint, true)
    }

    /// As [`Self::invoke`], for a preparation callback: a success is not counted as an encoding.
    #[doc(hidden)]
    pub fn prepare<T: GpuPainterBackend>(
        &self,
        prepare: impl FnOnce(&mut T) -> Result<()>,
    ) -> Result<()> {
        self.invoke_callback(prepare, false)
    }

    /// The draw mode the painter's adapter fixed at registration.
    #[doc(hidden)]
    pub fn draw_mode(&self) -> GpuDrawMode {
        self.state.lock().painter.draw_mode()
    }

    fn invoke_callback<T: GpuPainterBackend>(
        &self,
        callback: impl FnOnce(&mut T) -> Result<()>,
        records_encoding: bool,
    ) -> Result<()> {
        let mut state = self.state.lock();
        let painter = (state.painter.as_mut() as &mut dyn Any).downcast_mut::<T>();
        let result = match painter {
            Some(painter) => callback(painter),
            None => Err(anyhow::anyhow!(
                "GPU painter is incompatible with this backend"
            )),
        };
        match &result {
            Ok(()) if records_encoding => {
                state.status.encoded = state.status.encoded.saturating_add(1);
            }
            Err(error) => state.status.last_error = Some(format!("{error:#}")),
            Ok(()) => {}
        }
        result
    }
}

/// A weak painter registration: the registry never keeps a dropped painter alive.
#[doc(hidden)]
pub struct GpuPainterRegistration(Weak<Mutex<PainterState>>);

impl GpuPainterRegistration {
    fn reset(&self, reason: GpuResetReason) {
        if let Some(state) = self.0.upgrade() {
            let mut state = state.lock();
            state.painter.reset(reason);
            state.status.resets = state.status.resets.saturating_add(1);
        }
    }
}

/// The painters registered on a window, told when its device is retired, even before their first paint.
#[doc(hidden)]
#[derive(Default)]
pub struct GpuPainterRegistry(Mutex<Vec<GpuPainterRegistration>>);

impl GpuPainterRegistry {
    /// Registers a painter without extending its lifetime.
    pub fn register(&self, registration: GpuPainterRegistration) {
        let mut registrations = self.0.lock();
        registrations.retain(|entry| entry.0.strong_count() > 0);
        registrations.push(registration);
    }

    /// Notify before retiring a device or the window. Do not reenter this registry.
    pub fn reset(&self, reason: GpuResetReason) {
        let mut registrations = self.0.lock();
        for registration in registrations.iter() {
            registration.reset(reason);
        }
        if reason == GpuResetReason::WindowDestroyed {
            registrations.clear();
        }
    }
}

impl Drop for GpuPainterRegistry {
    fn drop(&mut self) {
        self.reset(GpuResetReason::WindowDestroyed);
    }
}

/// A painter and application snapshot retained by the scene.
#[derive(Clone)]
pub struct GpuPaintPrimitive {
    /// Window-scoped painter.
    pub handle: AnyGpuPainterHandle,
    /// Application snapshot.
    pub data: Arc<dyn Any + Send + Sync>,
    /// Logical to physical scale at scene construction.
    pub scale_factor: f32,
    /// Inherited element opacity.
    pub opacity: f32,
}

impl fmt::Debug for GpuPaintPrimitive {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuPaintPrimitive")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl GpuPaintPrimitive {
    /// The effective physical clip for the actual rendering target.
    #[doc(hidden)]
    pub fn target(
        &self,
        size: [u32; 2],
        bounds: Bounds<ScaledPixels>,
        mask: ContentMask<ScaledPixels>,
        sample_count: u32,
    ) -> GpuPaintTarget {
        let target_bounds = Bounds {
            origin: crate::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: crate::size(ScaledPixels(size[0] as f32), ScaledPixels(size[1] as f32)),
        };
        GpuPaintTarget {
            size,
            bounds,
            clip: bounds.intersect(&mask.bounds).intersect(&target_bounds),
            scale_factor: self.scale_factor,
            sample_count,
            opacity: self.opacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AnyGpuPainterHandle, GpuPaintPrimitive, GpuPaintTarget, GpuPainterBackend, GpuPainterRegistry,
        GpuResetReason,
    };
    use crate::{
        Bounds, ContentMask, Context, GpuPaintSurface, IntoElement, PrimitiveBatch, Quad, Render,
        ScaledPixels, Scene, Window, WindowHandle, WindowId, div, point, size,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct TestView;

    impl Render for TestView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
        }
    }

    struct RecordingPainter(Arc<Mutex<Vec<GpuResetReason>>>);

    impl GpuPainterBackend for RecordingPainter {
        fn reset(&mut self, reason: GpuResetReason) {
            self.0.lock().push(reason);
        }
    }

    fn square(left: f32, top: f32, side: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(left), ScaledPixels(top)),
            size(ScaledPixels(side), ScaledPixels(side)),
        )
    }

    fn surface(handle: AnyGpuPainterHandle, bounds: Bounds<ScaledPixels>, snapshot: u32) -> GpuPaintSurface {
        GpuPaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            draw: GpuPaintPrimitive {
                handle,
                data: Arc::new(snapshot),
                scale_factor: 2.0,
                opacity: 0.5,
            },
        }
    }

    fn recording() -> (AnyGpuPainterHandle, super::GpuPainterRegistration, Arc<Mutex<Vec<GpuResetReason>>>) {
        let owner = WindowHandle::<TestView>::new(WindowId::default()).into();
        let reasons = Arc::new(Mutex::new(Vec::new()));
        let (handle, registration) =
            AnyGpuPainterHandle::from_box(owner, Box::new(RecordingPainter(reasons.clone())));
        (handle, registration, reasons)
    }

    #[test]
    fn scissor_rounds_inward_and_clamps_to_the_target() {
        let mut target = GpuPaintTarget {
            size: [100, 80],
            bounds: Bounds::default(),
            clip: Bounds::new(
                point(ScaledPixels(-5.5), ScaledPixels(4.5)),
                size(ScaledPixels(110.0), ScaledPixels(100.0)),
            ),
            scale_factor: 2.0,
            sample_count: 1,
            opacity: 1.0,
        };
        assert_eq!(target.scissor_rect(), [0, 5, 100, 75]);
        target.clip = Bounds::new(
            point(ScaledPixels(10.2), ScaledPixels(12.2)),
            size(ScaledPixels(0.5), ScaledPixels(0.5)),
        );
        assert_eq!(target.scissor_rect(), [11, 13, 0, 0]);
    }

    #[test]
    fn a_gpu_paint_is_batched_in_draw_order_between_the_quads_around_it() {
        let (handle, _registration, _) = recording();
        let bounds = square(0.0, 0.0, 100.0);
        let quad = Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        };
        let mut scene = Scene::default();
        scene.insert_primitive(quad.clone());
        scene.insert_primitive(surface(handle, bounds, 1));
        scene.insert_primitive(quad);
        scene.finish();
        let batches: Vec<_> = scene.batches().collect();
        assert!(
            matches!(
                batches.as_slice(),
                [
                    PrimitiveBatch::Quads(_),
                    PrimitiveBatch::GpuPaints(_),
                    PrimitiveBatch::Quads(_),
                ]
            ),
            "{batches:?}"
        );
    }

    #[test]
    fn a_replayed_scene_keeps_the_painter_and_its_snapshot_alive() {
        let (handle, registration, reasons) = recording();
        let registry = GpuPainterRegistry::default();
        registry.register(registration);
        let mut scene = Scene::default();
        scene.insert_primitive(surface(handle, square(0.0, 0.0, 100.0), 42));
        scene.finish();
        let weak = Arc::downgrade(&scene.gpu_paints[0].draw.data);
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        drop(scene);
        assert_eq!(replay.gpu_paints.len(), 1);
        assert!(weak.upgrade().is_some());
        replay.clear();
        assert!(weak.upgrade().is_none());
        registry.reset(GpuResetReason::DeviceReplaced);
        drop(registry);
        assert_eq!(*reasons.lock(), [GpuResetReason::DeviceReplaced]);
    }

    #[test]
    fn a_fully_clipped_gpu_paint_is_not_inserted() {
        let (handle, _registration, _) = recording();
        let mut clipped = surface(handle, square(200.0, 200.0, 10.0), 1);
        clipped.content_mask = ContentMask {
            bounds: square(0.0, 0.0, 100.0),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(clipped);
        assert!(scene.gpu_paints.is_empty());
    }

    #[test]
    fn the_registry_tells_a_painter_that_never_drew_once_the_window_goes() {
        let (handle, registration, reasons) = recording();
        let registry = GpuPainterRegistry::default();
        registry.register(registration);
        drop(registry);
        assert_eq!(*reasons.lock(), [GpuResetReason::WindowDestroyed]);
        assert_eq!(handle.status().resets, 1);
        assert_eq!(handle.status().encoded, 0);
    }

    #[test]
    fn dispatch_checks_the_adapter_type_and_records_results() {
        struct OtherPainter;
        impl GpuPainterBackend for OtherPainter {
            fn reset(&mut self, _: GpuResetReason) {}
        }
        let (handle, _registration, _) = recording();
        assert!(handle.invoke::<OtherPainter>(|_| Ok(())).is_err());
        assert_eq!(handle.status().encoded, 0);
        assert!(handle.invoke::<RecordingPainter>(|_| anyhow::bail!("encoding failed")).is_err());
        assert!(handle.invoke::<RecordingPainter>(|_| Ok(())).is_ok());
        assert!(handle.prepare::<RecordingPainter>(|_| Ok(())).is_ok());
        assert_eq!(handle.status().encoded, 1);
        assert_eq!(handle.take_error().as_deref(), Some("encoding failed"));
        assert!(handle.take_error().is_none());
    }

    #[test]
    fn the_target_clip_is_the_element_the_parent_and_the_target() {
        let (handle, _registration, _) = recording();
        let draw = surface(handle, square(0.0, 0.0, 1.0), 0).draw;
        let bounds = Bounds {
            origin: point(ScaledPixels(-10.0), ScaledPixels(20.0)),
            size: size(ScaledPixels(80.0), ScaledPixels(80.0)),
        };
        let mask = ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(5.0), ScaledPixels(10.0)),
                size: size(ScaledPixels(100.0), ScaledPixels(60.0)),
            },
        };
        let target = draw.target([50, 60], bounds, mask, 1);
        assert_eq!(target.bounds, bounds);
        assert_eq!(target.clip.origin, point(ScaledPixels(5.0), ScaledPixels(20.0)));
        assert_eq!(target.clip.size, size(ScaledPixels(45.0), ScaledPixels(40.0)));
        assert_eq!((target.scale_factor, target.opacity), (2.0, 0.5));
    }
}
