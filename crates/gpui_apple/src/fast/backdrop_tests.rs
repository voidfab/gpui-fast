//! vulpine (FOX9-a010dd): backdrop blur through the headless Metal renderer, read back. Skathi's tests ported to this
//! fork (Apache-2.0), plus the CPU twin the readback is held to and the allocation rule.

use super::{FORCE_INTERMEDIATE, FULL_FRAME, SCRATCH_RELEASE_FRAMES};
use crate::metal_renderer::{InstanceBufferPool, MetalRenderer};
use anyhow::{Context as _, Result};
use gpui::{
    BackdropBlurRect, Bounds, ContentMask, Corners, DevicePixels, Hsla, Quad, Rgba, ScaledPixels,
    Scene, Size, point, rgb, size,
};
use parking_lot::Mutex;
use std::sync::Arc;

/// The readback against its CPU twin: no channel of any pixel more than this many 8-bit steps apart. The GPU and the
/// twin take the same taps at the same weights (every bilinear weight here is 0, 1/4, 1/2 or 3/4, exact in any filter
/// precision) and both round to 8 bits after each pass; what is left is the GPU's own rounding of a pass's sum, at
/// most a step per pass over three passes and the composite's read. Fixed before the first run; never loosened to pass.
const TWIN_MAX_DIFF: u8 = 3;
/// ... and the mean absolute difference over the blurred rect, in 8-bit steps.
const TWIN_MEAN_DIFF: f64 = 1.0;

fn renderer() -> Result<MetalRenderer> {
    MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
}

fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn mask(width: f32, height: f32) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds(0., 0., width, height),
    }
}

fn quad(rect: Bounds<ScaledPixels>, color: Rgba, frame: f32) -> Quad {
    Quad {
        bounds: rect,
        background: color.into(),
        content_mask: mask(frame, frame),
        ..Default::default()
    }
}

fn frame_size(side: f32) -> Size<DevicePixels> {
    size(DevicePixels(side as i32), DevicePixels(side as i32))
}

fn render(renderer: &mut MetalRenderer, mut scene: Scene, side: f32) -> Result<image::RgbaImage> {
    scene.finish();
    renderer.render_scene_to_image(&scene, frame_size(side))
}

fn pixel(image: &image::RgbaImage, x: u32, y: u32) -> [u8; 4] {
    image.get_pixel(x, y).0
}

/// Black, with white cells of `cell` pixels in a checkerboard.
fn checkerboard(side: f32, cell: f32) -> Scene {
    let mut scene = Scene::default();
    scene.insert_primitive(quad(bounds(0., 0., side, side), rgb(0x000000), side));
    let cells = (side / cell) as usize;
    for row in 0..cells {
        for column in 0..cells {
            if (row + column).is_multiple_of(2) {
                let rect = bounds(column as f32 * cell, row as f32 * cell, cell, cell);
                scene.insert_primitive(quad(rect, rgb(0xffffff), side));
            }
        }
    }
    scene
}

fn checker_value(x: usize, y: usize, cell: usize) -> f32 {
    if (x / cell + y / cell).is_multiple_of(2) {
        1.
    } else {
        0.
    }
}

/// One channel of a frame or a pyramid level, values in 0..=1, as an 8-bit unorm texture holds them.
#[derive(Clone)]
struct Plane {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl Plane {
    fn texel(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        self.values[y * self.width + x]
    }

    /// A linear, clamp-to-edge sample at normalised `(u, v)`, as the shaders' sampler takes it.
    fn sample(&self, u: f32, v: f32) -> f32 {
        let tx = u * self.width as f32 - 0.5;
        let ty = v * self.height as f32 - 0.5;
        let (x0, y0) = (tx.floor(), ty.floor());
        let (fx, fy) = (tx - x0, ty - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = self.texel(x0, y0) * (1. - fx) + self.texel(x0 + 1, y0) * fx;
        let bottom = self.texel(x0, y0 + 1) * (1. - fx) + self.texel(x0 + 1, y0 + 1) * fx;
        top * (1. - fy) + bottom * fy
    }

    /// A pass into a `width` x `height` target, each texel rounded to 8 bits as the level stores it.
    fn pass(&self, width: usize, height: usize, taps: &[(f32, f32, f32)]) -> Plane {
        let texel = (1. / self.width as f32, 1. / self.height as f32);
        let mut values = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let sum: f32 = taps
                    .iter()
                    .map(|&(dx, dy, weight)| {
                        self.sample(u + texel.0 * dx, v + texel.1 * dy) * weight
                    })
                    .sum();
                values.push(quantize(sum));
            }
        }
        Plane {
            width,
            height,
            values,
        }
    }
}

fn quantize(value: f32) -> f32 {
    (value.clamp(0., 1.) * 255.).round() / 255.
}

const DOWN: [(f32, f32, f32); 5] = [
    (0., 0., 0.5),
    (-1., -1., 0.125),
    (1., -1., 0.125),
    (-1., 1., 0.125),
    (1., 1., 0.125),
];
const UP: [(f32, f32, f32); 8] = [
    (-1., 0., 0.125),
    (1., 0., 0.125),
    (0., -1., 0.125),
    (0., 1., 0.125),
    (-1., -1., 0.125),
    (1., -1., 0.125),
    (-1., 1., 0.125),
    (1., 1., 0.125),
];

/// The twin of the shaders' pyramid: `levels` halvings down, then back up to level 0 (half the frame).
fn cpu_pyramid(frame: &Plane, levels: usize) -> Plane {
    let mut pyramid = vec![frame.clone()];
    for level in 1..=levels {
        let source = &pyramid[level - 1];
        let next = source.pass(
            (frame.width >> level).max(1),
            (frame.height >> level).max(1),
            &DOWN,
        );
        pyramid.push(next);
    }
    let mut up = pyramid[levels].clone();
    for level in (1..levels).rev() {
        let target = &pyramid[level];
        up = up.pass(target.width, target.height, &UP);
    }
    up
}

#[test]
fn a_blurred_rect_over_a_checkerboard_matches_its_cpu_twin() -> Result<()> {
    const SIDE: f32 = 128.;
    const CELL: usize = 8;
    let rect = bounds(16., 16., 96., 96.);
    let mut renderer = renderer()?;
    let mut scene = checkerboard(SIDE, CELL as f32);
    scene.insert_primitive(BackdropBlurRect {
        bounds: rect,
        content_mask: mask(SIDE, SIDE),
        blur_radius: ScaledPixels(10.),
        ..Default::default()
    });
    let image = render(&mut renderer, scene, SIDE)?;

    let side = SIDE as usize;
    let frame = Plane {
        width: side,
        height: side,
        values: (0..side * side)
            .map(|i| checker_value(i % side, i / side, CELL))
            .collect(),
    };
    // A 10 px blur is two pyramid levels.
    let level0 = cpu_pyramid(&frame, 2);
    let (mut worst, mut total, mut count) = (0u8, 0u64, 0u64);
    for y in 0..side {
        for x in 0..side {
            let actual = pixel(&image, x as u32, y as u32);
            let inside = (16..112).contains(&x) && (16..112).contains(&y);
            if !inside {
                let expected = (checker_value(x, y, CELL) * 255.) as u8;
                assert_eq!(
                    actual,
                    [expected, expected, expected, 255],
                    "outside the rect the frame is untouched, at ({x}, {y})"
                );
                continue;
            }
            let u = (x as f32 + 0.5) / SIDE;
            let v = (y as f32 + 0.5) / SIDE;
            let expected = (quantize(level0.sample(u, v)) * 255.).round() as u8;
            for channel in &actual[..3] {
                let diff = channel.abs_diff(expected);
                worst = worst.max(diff);
                total += u64::from(diff);
                count += 1;
            }
            assert_eq!(actual[3], 255, "opaque at ({x}, {y})");
        }
    }
    let mean = total as f64 / count as f64;
    println!(
        "checkerboard twin: worst {worst}, mean {mean:.4} (8-bit steps) over {count} channels"
    );
    assert!(
        worst <= TWIN_MAX_DIFF,
        "the readback strays {worst} steps from its CPU twin (at most {TWIN_MAX_DIFF})"
    );
    assert!(
        mean <= TWIN_MEAN_DIFF,
        "the readback strays {mean:.4} steps on average from its CPU twin (at most {TWIN_MEAN_DIFF})"
    );
    // And it is a blur: the checkerboard's contrast is gone in the middle of the rect.
    let middle: Vec<u8> = (40..88).map(|x| pixel(&image, x, 64)[0]).collect();
    let (low, high) = (middle.iter().min().copied(), middle.iter().max().copied());
    assert!(
        high.zip(low).is_some_and(|(high, low)| high - low < 128),
        "the blur must soften the cells: {low:?}..{high:?}"
    );
    Ok(())
}

#[test]
fn a_scene_without_blur_rects_allocates_nothing() -> Result<()> {
    const SIDE: f32 = 64.;
    let mut renderer = renderer()?;
    render(&mut renderer, checkerboard(SIDE, 8.), SIDE)?;
    assert!(
        !renderer.backdrop.has_pipelines(),
        "no backdrop pipeline before a blur"
    );
    assert_eq!(
        renderer.backdrop.held_levels(),
        None,
        "no scratch before a blur"
    );
    assert_eq!(renderer.backdrop.scratch_bytes(), 0);

    let mut scene = checkerboard(SIDE, 8.);
    scene.insert_primitive(BackdropBlurRect {
        bounds: bounds(8., 8., 48., 48.),
        content_mask: mask(SIDE, SIDE),
        blur_radius: ScaledPixels(10.),
        ..Default::default()
    });
    render(&mut renderer, scene, SIDE)?;
    assert_eq!(
        renderer.backdrop.held_levels(),
        Some(2),
        "a 10 px blur allocates its two levels and no more"
    );
    assert!(renderer.backdrop.scratch_bytes() > 0);

    // The scratch stays through short gaps and goes after SCRATCH_RELEASE_FRAMES frames without a blur.
    for frame in 1..SCRATCH_RELEASE_FRAMES {
        render(&mut renderer, checkerboard(SIDE, 8.), SIDE)?;
        assert_eq!(
            renderer.backdrop.held_levels(),
            Some(2),
            "kept through frame {frame} without a blur"
        );
    }
    render(&mut renderer, checkerboard(SIDE, 8.), SIDE)?;
    assert_eq!(renderer.backdrop.held_levels(), None, "released");
    assert_eq!(renderer.backdrop.scratch_bytes(), 0);
    Ok(())
}

fn split_scene(side: f32) -> Scene {
    let mut scene = Scene::default();
    scene.insert_primitive(quad(bounds(0., 0., side, side), rgb(0x000000), side));
    scene.insert_primitive(quad(bounds(0., 0., side / 2., side), rgb(0xffffff), side));
    scene
}

#[test]
fn corners_mask_tint_and_opacity() -> Result<()> {
    const SIDE: f32 = 64.;
    let mut renderer = renderer()?;
    let blur = |tint: Hsla, opacity: f32| BackdropBlurRect {
        bounds: bounds(8., 8., 48., 48.),
        corner_radii: Corners::all(ScaledPixels(12.)),
        content_mask: mask(40., SIDE),
        blur_radius: ScaledPixels(10.),
        tint,
        opacity,
        ..Default::default()
    };
    let red = Hsla {
        a: 0.5,
        ..gpui::red()
    };
    let mut scene = split_scene(SIDE);
    scene.insert_primitive(blur(red, 1.));
    let tinted = render(&mut renderer, scene, SIDE)?;
    assert_eq!(
        pixel(&tinted, 9, 9),
        [255; 4],
        "outside the rounded corner: crisp, untouched"
    );
    assert_eq!(
        pixel(&tinted, 50, 32),
        [0, 0, 0, 255],
        "outside the content mask"
    );
    let dark = pixel(&tinted, 36, 32);
    assert!(
        dark[0] > dark[1] + 100 && dark[1] == dark[2],
        "a half-alpha red tint over the dark side: {dark:?}"
    );

    let mut scene = split_scene(SIDE);
    scene.insert_primitive(blur(Hsla::transparent_black(), 1.));
    let opaque = render(&mut renderer, scene, SIDE)?;
    let mut scene = split_scene(SIDE);
    scene.insert_primitive(blur(Hsla::transparent_black(), 0.5));
    let half = render(&mut renderer, scene, SIDE)?;
    let (full, faded) = (pixel(&opaque, 33, 32)[0], pixel(&half, 33, 32)[0]);
    assert!(full > 16, "the blur reaches the black side: {full}");
    assert!(
        faded.abs_diff(full / 2) <= 2,
        "opacity 0.5 lands halfway between the frame and the blur: {faded} vs {full}"
    );
    Ok(())
}

#[test]
fn a_small_blur_processes_its_region_only_and_matches_the_whole_frame() -> Result<()> {
    const SIDE: f32 = 256.;
    let mut renderer = renderer()?;
    let scene = || {
        let mut scene = checkerboard(SIDE, 8.);
        scene.insert_primitive(BackdropBlurRect {
            bounds: bounds(100., 100., 40., 30.),
            content_mask: mask(SIDE, SIDE),
            blur_radius: ScaledPixels(15.),
            ..Default::default()
        });
        scene
    };
    let region = render(&mut renderer, scene(), SIDE)?;
    FULL_FRAME.set(true);
    let whole = render(&mut renderer, scene(), SIDE);
    FULL_FRAME.set(false);
    let whole = whole?;
    assert_eq!(
        region.as_raw(),
        whole.as_raw(),
        "the region's blur is the whole frame's"
    );
    Ok(())
}

#[test]
fn the_intermediate_frame_reproduces_the_direct_render() -> Result<()> {
    const SIDE: f32 = 64.;
    let mut renderer = renderer()?;
    let scene = || {
        let mut scene = split_scene(SIDE);
        scene.insert_primitive(BackdropBlurRect {
            bounds: bounds(8., 8., 48., 48.),
            corner_radii: Corners::all(ScaledPixels(8.)),
            content_mask: mask(SIDE, SIDE),
            blur_radius: ScaledPixels(10.),
            ..Default::default()
        });
        scene.insert_primitive(quad(bounds(20., 20., 8., 8.), rgb(0x00ff00), SIDE));
        scene
    };
    let direct = render(&mut renderer, scene(), SIDE)?;
    FORCE_INTERMEDIATE.set(true);
    let through = render(&mut renderer, scene(), SIDE);
    FORCE_INTERMEDIATE.set(false);
    assert_eq!(direct.as_raw(), through?.as_raw());
    assert_eq!(
        pixel(&direct, 24, 24),
        [0, 255, 0, 255],
        "a later quad paints above the blur"
    );
    Ok(())
}

#[test]
fn inside_a_layer_what_is_painted_after_the_blur_stays_above_it() -> Result<()> {
    const SIDE: f32 = 64.;
    let mut renderer = renderer()?;
    let mut scene = Scene::default();
    scene.push_layer(bounds(0., 0., SIDE, SIDE));
    scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000), SIDE));
    scene.insert_primitive(quad(bounds(0., 0., SIDE / 2., SIDE), rgb(0xffffff), SIDE));
    scene.insert_primitive(BackdropBlurRect {
        bounds: bounds(8., 8., 48., 48.),
        content_mask: mask(SIDE, SIDE),
        blur_radius: ScaledPixels(10.),
        ..Default::default()
    });
    // A card's own content, painted after its glass in the same layer.
    scene.insert_primitive(quad(bounds(36., 36., 8., 8.), rgb(0x0000ff), SIDE));
    scene.pop_layer();
    let image = render(&mut renderer, scene, SIDE)?;
    assert_eq!(
        pixel(&image, 40, 40),
        [0, 0, 255, 255],
        "the later quad is above the blur"
    );
    let softened = pixel(&image, 30, 20)[0];
    assert!(
        (17..240).contains(&softened),
        "the earlier quads are under it, blurred: {softened}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------- measurement (ignored)

fn offscreen(renderer: &MetalRenderer, width: usize, height: usize) -> Result<super::Texture> {
    use objc2_metal::{MTLDevice, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage};
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: Both dimensions are positive and this descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(width);
        descriptor.setHeight(height);
    }
    descriptor.setPixelFormat(renderer.pixel_format_for_tests());
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    renderer
        .device
        .newTextureWithDescriptor(&descriptor)
        .context("allocating the measurement target")
}

/// One frame's GPU time, ms: the command buffer's GPUEndTime - GPUStartTime (GPU timestamps).
fn gpu_ms(renderer: &mut MetalRenderer, scene: &Scene, width: usize, height: usize) -> Result<f64> {
    use objc2::msg_send;
    use objc2_metal::MTLCommandBuffer;
    let size = size(DevicePixels(width as i32), DevicePixels(height as i32));
    objc2::rc::autoreleasepool(|_| {
        renderer.update_path_intermediate_textures(size)?;
        let target = offscreen(renderer, width, height)?;
        let commands = renderer.render_frame(scene, &target, size)?;
        commands.commit();
        commands.waitUntilCompleted();
        // SAFETY: plain CFTimeInterval reads of a completed command buffer.
        let start: f64 = unsafe { msg_send![&*commands, GPUStartTime] };
        let end: f64 = unsafe { msg_send![&*commands, GPUEndTime] };
        Ok((end - start) * 1000.)
    })
}

fn stats(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2], samples[samples.len() * 9 / 10])
}

/// A busy 2x background: stripes and cells, so the pyramid reads real contrast.
fn backdrop_scene(width: f32, height: f32) -> Scene {
    let mut scene = Scene::default();
    let frame = ContentMask {
        bounds: bounds(0., 0., width, height),
    };
    let fill = |rect, color: Rgba| Quad {
        bounds: rect,
        background: color.into(),
        content_mask: frame,
        ..Default::default()
    };
    scene.insert_primitive(fill(bounds(0., 0., width, height), rgb(0x2a5d3a)));
    let mut x = 0.;
    let mut i = 0u32;
    while x < width {
        let color = if i.is_multiple_of(2) {
            rgb(0xc8d870)
        } else {
            rgb(0x6fa05a)
        };
        scene.insert_primitive(fill(bounds(x, height * 0.4, 6., height * 0.6), color));
        x += 14.;
        i += 1;
    }
    scene
}

/// GLS1's numbers: the GPU ms one frosted card adds at 2x, at two window sizes, with and without the intermediate a
/// framebuffer-only drawable needs, and the scratch it holds while on screen. Run alone:
/// `cargo test -p gpui_apple --lib fast::backdrop::tests::gpu_ms_per_blur_at_2x -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn gpu_ms_per_blur_at_2x() -> Result<()> {
    const WARMUP: usize = 30;
    const SAMPLES: usize = 120;
    let mut renderer = renderer()?;
    // (name, frame w, h, card x, y, w, h, radius in device px)
    let cases = [
        (
            "desktop-1440x900@2x",
            2880.,
            1800.,
            480.,
            300.,
            1920.,
            1200.,
            20.,
        ),
        (
            "fox-picture-418x264@2x",
            836.,
            528.,
            64.,
            96.,
            708.,
            336.,
            16.,
        ),
    ];
    for (name, width, height, cx, cy, cw, ch, radius) in cases {
        let plain = {
            let mut scene = backdrop_scene(width, height);
            scene.finish();
            scene
        };
        let glass = {
            let mut scene = backdrop_scene(width, height);
            scene.insert_primitive(BackdropBlurRect {
                bounds: bounds(cx, cy, cw, ch),
                content_mask: ContentMask {
                    bounds: bounds(0., 0., width, height),
                },
                corner_radii: Corners::all(ScaledPixels(24.)),
                blur_radius: ScaledPixels(radius),
                tint: Hsla {
                    h: 0.6,
                    s: 0.2,
                    l: 0.95,
                    a: 0.35,
                },
                ..Default::default()
            });
            scene.finish();
            scene
        };
        let (w, h) = (width as usize, height as usize);
        let mut run = |scene: &Scene, intermediate: bool| -> Result<(f64, f64, usize)> {
            FORCE_INTERMEDIATE.set(intermediate);
            for _ in 0..WARMUP {
                gpu_ms(&mut renderer, scene, w, h)?;
            }
            let samples = (0..SAMPLES)
                .map(|_| gpu_ms(&mut renderer, scene, w, h))
                .collect::<Result<Vec<_>>>()?;
            let bytes = renderer.backdrop.scratch_bytes();
            FORCE_INTERMEDIATE.set(false);
            let (p50, p90) = stats(samples);
            Ok((p50, p90, bytes))
        };
        let (plain_p50, plain_p90, _) = run(&plain, false)?;
        let (glass_p50, glass_p90, glass_bytes) = run(&glass, false)?;
        let (inter_p50, inter_p90, inter_bytes) = run(&glass, true)?;
        let levels = BackdropBlurRect {
            blur_radius: ScaledPixels(radius),
            ..Default::default()
        }
        .effective_kernel_levels();
        println!(
            "gpu-ms {name}: frame {w}x{h}, card {cw}x{ch} r{radius} ({levels} levels); plain p50 {plain_p50:.3} p90 {plain_p90:.3}; \
             glass p50 {glass_p50:.3} p90 {glass_p90:.3} (blur +{:.3} ms p50); glass via intermediate p50 {inter_p50:.3} p90 {inter_p90:.3} \
             (+{:.3} ms p50); scratch {:.1} MB direct, {:.1} MB with the intermediate; samples {SAMPLES} after {WARMUP} warm-up",
            glass_p50 - plain_p50,
            inter_p50 - plain_p50,
            glass_bytes as f64 / 1e6,
            inter_bytes as f64 / 1e6,
        );
    }
    Ok(())
}
