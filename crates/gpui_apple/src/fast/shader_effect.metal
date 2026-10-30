// vulpine (FOX9-894bf2), not upstream: the shader effect's pass (shader_effect.rs).
// Compiled at runtime the first time a frame draws an effect; the published
// shader library is left as it is.
//
// The ASCII look derives from Vigil's web pass (apps/vigil/src/effects/asciiShader.ts):
// nine 5x5 glyphs (25-bit masks, opentouchdesigner's ascii.json, MIT; reordered by
// coverage, with a denser final mask), in a 5x7 terminal cell. Four local mip taps
// keep thin structures rather than averaging the whole cell away. Rec.709 luma
// above the ground threshold picks the glyph; phosphor blends toward source color.
// CRT preserves the picture: restrained glass, channel separation, aperture grille,
// scanlines, corner falloff and highlight-only bloom. Its band runs only on entry.

#include <metal_stdlib>
using namespace metal;

struct VulpineEffect {
  float4 region;   // the copy's place in the target: x, y, width, height (pixels)
  float4 anchor;   // the effect's bounds: x, y, width, height (the grid starts at x, y)
  float4 look;     // kind (1 ASCII, 2 CRT), mix, cell (pixels), tint
  float4 tone;     // curve, level, scanlines, phase (the CRT band's place, -1..2)
  float4 ground;   // red, green, blue, threshold (ASCII: luma under it draws ground)
  float4 crt;      // curvature, aberration (pixels), vignette, glow
  float2 viewport; // the target's size in pixels
  float2 pad;
};

// gpui-fast's rounded clip stack, as its fragments read it (rounded_clip_shader.rs).
struct FastRoundedClip {
  float4 bounds;
  float4 horizontal;
  float4 vertical;
};

static float clip_coverage(float2 position, constant FastRoundedClip &shape) {
  float2 extent = shape.bounds.zw - shape.bounds.xy;
  if (extent.x <= 0 || extent.y <= 0) return 0;
  float2 local = position - shape.bounds.xy;
  float distance = max(max(-local.x, -local.y), max(local.x - extent.x, local.y - extent.y));
  for (uint corner = 0; corner < 4; ++corner) {
    float2 radius = float2(shape.horizontal[corner], shape.vertical[corner]);
    if (radius.x <= 0 || radius.y <= 0) continue;
    bool right = corner == 1 || corner == 2;
    bool bottom = corner >= 2;
    float2 inward = float2(right ? extent.x - local.x : local.x, bottom ? extent.y - local.y : local.y);
    if (inward.x < radius.x && inward.y < radius.y) {
      float2 q = (inward - radius) / radius;
      float length_q = length(q);
      float gradient = length(q / radius) / max(length_q, 0.000001f);
      distance = max(distance, (length_q - 1) / max(gradient, 0.000001f));
    }
  }
  return clamp(0.5f - distance, 0.0f, 1.0f);
}

struct EffectVertex {
  float4 position [[position]];
};

vertex EffectVertex vulpine_effect_vertex(uint id [[vertex_id]],
                                          constant VulpineEffect &e [[buffer(0)]]) {
  // Two triangles over the region: (0,0) (1,0) (0,1), (0,1) (1,0) (1,1).
  float2 unit = float2((id == 1 || id == 4 || id == 5) ? 1.0 : 0.0,
                       (id == 2 || id == 3 || id == 5) ? 1.0 : 0.0);
  float2 pixel = e.region.xy + unit * e.region.zw;
  EffectVertex out;
  out.position = float4(pixel.x / e.viewport.x * 2.0 - 1.0, 1.0 - pixel.y / e.viewport.y * 2.0, 0.0, 1.0);
  return out;
}

constexpr sampler cells(filter::linear, mip_filter::linear, address::clamp_to_edge);
constexpr sampler pixels(filter::linear, address::clamp_to_edge);

constant int GLYPHS[9] = {4, 131200, 14336, 145536, 18157905, 1016800, 27070835, 11512810, 33492799};
constant float3 PHOSPHOR = float3(0.25, 1.0, 0.45);

// One glyph pixel; local is the cell in [0,1)^2, y down; bit (4 - col) + 5 * (4 - row).
static float glyph_bit(int bits, float2 local) {
  float2 g = float2((local.x - 0.5) * 1.1 + 0.5, (local.y * 7.0 - 1.0) / 5.0);
  int col = int(floor(g.x * 5.0));
  int row = int(floor(g.y * 5.0));
  if (col < 0 || row < 0 || col > 4 || row > 4) return 0.0;
  return float((bits >> ((4 - col) + 5 * (4 - row))) & 1);
}

static float2 copy_uv(constant VulpineEffect &e, float2 pixel) {
  return (pixel - e.region.xy) / e.region.zw;
}

static float3 ascii(constant VulpineEffect &e, texture2d<float> copy, float2 frag) {
  float cell = max(e.look.z, 2.0);
  float2 pitch = float2(cell, cell * 1.4);
  float2 local = frag - e.anchor.xy;
  float2 index = floor(local / pitch);
  float2 center = e.anchor.xy + (index + 0.5) * pitch;
  float2 offset = pitch * 0.25;
  float lod = max(0.0, log2(cell * 0.5));
  float4 src = float4(0.0);
  for (uint tap = 0; tap < 4; ++tap) {
    float2 sign = float2((tap & 1) ? 1.0 : -1.0, (tap & 2) ? 1.0 : -1.0);
    float2 at = clamp(center + sign * offset, e.anchor.xy + 0.5, e.anchor.xy + e.anchor.zw - 0.5);
    src += copy.sample(cells, copy_uv(e, at), level(lod)) * 0.25;
  }
  float gain = 1.0 + 0.4 * clamp(e.tone.y, 0.0, 1.0);
  float luma = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
  float l = pow(clamp(luma * gain, 0.0, 1.0), max(e.tone.x, 0.05));
  if (l < e.ground.w) return e.ground.rgb;
  float density = clamp((l - e.ground.w) / max(1.0 - e.ground.w, 0.001), 0.0, 1.0);
  int glyph = int(clamp(ceil(density * 9.0) - 1.0, 0.0, 8.0));
  float on = glyph_bit(GLYPHS[glyph], (local - index * pitch) / pitch);
  float scan = 1.0 - e.tone.z * 0.55 * (1.0 - step(0.5, fract(frag.y * 0.5)));
  float3 color = mix(PHOSPHOR, src.rgb * 1.4, clamp(e.look.w, 0.0, 1.0)) * min(gain, 1.25);
  return mix(e.ground.rgb, color, on * scan);
}

static float3 crt(constant VulpineEffect &e, texture2d<float> copy, float2 frag) {
  float2 p = (frag - e.anchor.xy) / max(e.anchor.zw, float2(1.0));
  // The glass: points further from the center read from further out, so the picture bulges.
  float2 q = p * 2.0 - 1.0;
  float bend = clamp(e.crt.x, 0.0, 1.0) * 0.08;
  q *= 1.0 + bend * dot(q, q);
  float2 g = q * 0.5 + 0.5;
  if (any(g < 0.0) || any(g > 1.0)) return e.ground.rgb;
  float2 at = e.anchor.xy + g * e.anchor.zw;
  float split = e.crt.y;
  float3 color = float3(copy.sample(pixels, copy_uv(e, at + float2(split, 0.0)), level(0)).r,
                        copy.sample(pixels, copy_uv(e, at), level(0)).g,
                        copy.sample(pixels, copy_uv(e, at - float2(split, 0.0)), level(0)).b);
  // A bounded cross of bright neighbours, not a veil lifted over dark pixels.
  float3 bloom = float3(0.0);
  for (uint tap = 0; tap < 4; ++tap) {
    float2 offset = tap < 2 ? float2(tap == 0 ? -2.0 : 2.0, 0.0)
                            : float2(0.0, tap == 2 ? -2.0 : 2.0);
    float3 light = copy.sample(pixels, copy_uv(e, at + offset), level(0)).rgb;
    bloom += max(light - 0.55, float3(0.0)) * 0.25;
  }
  color += bloom * clamp(e.crt.w, 0.0, 1.0) * 0.3;
  float scanlines = clamp(e.tone.z, 0.0, 1.0);
  float scan = 1.0 - scanlines * 0.2 * (0.5 + 0.5 * cos(frag.y * 2.094395));
  uint column = uint(frag.x) % 3;
  float3 grille = column == 0 ? float3(1.0, 0.85, 0.85) : (column == 1 ? float3(0.85, 1.0, 0.85) : float3(0.85, 0.85, 1.0));
  color *= scan * mix(float3(1.0), grille, scanlines * 0.45);
  float corners = 16.0 * g.x * g.y * (1.0 - g.x) * (1.0 - g.y);
  color *= mix(1.0, pow(clamp(corners, 0.0, 1.0), 0.35), clamp(e.crt.z, 0.0, 1.0));
  float band = exp(-pow((g.y - e.tone.w) * 9.0, 2.0));
  color += band * clamp(e.crt.w, 0.0, 1.0) * 0.08 * float3(0.85, 1.0, 0.9);
  return color;
}

fragment float4 vulpine_effect_fragment(EffectVertex in [[stage_in]],
                                        constant VulpineEffect &e [[buffer(0)]],
                                        texture2d<float> copy [[texture(0)]],
                                        constant FastRoundedClip *fast_clip_shapes [[buffer(28)]],
                                        constant uint &fast_clip_enabled [[buffer(29)]]) {
  float2 frag = in.position.xy;
  float4 original = copy.sample(pixels, copy_uv(e, frag), level(0));
  int kind = int(e.look.x + 0.5);
  float3 shaded = kind == 1 ? ascii(e, copy, frag) : (kind == 2 ? crt(e, copy, frag) : original.rgb);
  float amount = clamp(e.look.y, 0.0, 1.0);
  float4 color = float4(mix(original.rgb, shaded, amount), mix(original.a, 1.0, amount));
  float coverage = 1.0;
  for (uint index = 0; index < fast_clip_enabled; ++index) {
    coverage = min(coverage, clip_coverage(frag, fast_clip_shapes[index]));
  }
  // Premultiplied: what the target held is already premultiplied (GPUI's blend leaves it so).
  return color * coverage;
}
