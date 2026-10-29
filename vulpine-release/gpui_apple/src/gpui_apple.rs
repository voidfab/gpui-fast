// Modified by Longbridge for gpui-fast.
#![cfg(any(target_os = "macos", target_os = "ios"))]
//! Shared Apple rendering, task dispatch, and system services for GPUI.
//!
//! This crate renders GPUI scenes directly with Metal on Apple platforms. It
//! owns GPU resources, text rendering, Keychain access, and thermal notifications
//! while leaving application lifecycle, windowing, and input to each platform backend.

mod fast;
mod dispatcher;
// vulpine (FOX9-f0a295): the in-frame GPU painter's Metal backend.
pub mod gpu_painter;
pub mod keychain;
mod metal_atlas;
pub mod metal_renderer;
pub mod thermal;

#[cfg(feature = "font-kit")]
mod open_type;
#[cfg(feature = "font-kit")]
mod text_system;

use objc2::MainThreadMarker;

pub use dispatcher::{AppleActivity, AppleDispatcher};
pub use gpu_painter::{
    MetalDrawContext, MetalPaintContext, MetalPainter, MetalPrepareContext, MetalSharedPainter,
    metal_painter, metal_shared_painter,
};
/// The `metal` crate the painter contexts are written in, so a painter names the same types.
pub use metal;
#[cfg(feature = "font-kit")]
pub use text_system::AppleTextSystem;
/// vulpine (FOX9-894bf2): the shader effect's frame GPU times, for its measurement runs.
#[cfg(feature = "vulpine-effect")]
pub use fast::shader_effect::take_frame_gpu_ms as vulpine_effect_frame_gpu_ms;

#[inline]
pub fn main_thread_marker() -> Option<MainThreadMarker> {
    thread_local! {
        static MAIN_THREAD_MARKER: Option<MainThreadMarker> = MainThreadMarker::new();
    }
    MAIN_THREAD_MARKER.with(|marker| *marker)
}

#[cfg(test)]
mod tests {
    use super::main_thread_marker;

    #[test]
    fn main_thread_marker_rejects_background_threads() {
        std::thread::spawn(|| {
            assert_eq!(main_thread_marker(), None);
            assert_eq!(main_thread_marker(), None);
        })
        .join()
        .expect("background thread panicked");
    }
}
