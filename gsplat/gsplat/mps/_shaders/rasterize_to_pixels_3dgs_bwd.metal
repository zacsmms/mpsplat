// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3DGS backward tile rasterizer.
//
// Walks the same gaussians as the forward kernel, in reverse depth order,
// recomputes alpha to match forward, derives per-pixel gradient
// contributions, and atomic-adds them into per-gaussian gradient buffers.
//
// Differences from CUDA:
//   * No warp-leader reduction: every active thread does its own
//     `atomic_fetch_add` against `v_means2d` / `v_conics` / `v_colors` /
//     `v_opacities`. Slower than the CUDA scheme, but trivially correct
//     and a clean baseline to optimise from.
//   * No threadgroup cache yet — each thread reads gaussian state directly
//     from device memory. Stage 7.x can add a cache once the baseline is
//     verified to match the torch reference within tolerance.
//   * `MAX_ALPHA` and `ALPHA_THRESHOLD` semantics match the *forward*
//     kernel's relaxed flavour (no alpha-threshold cull, no early-exit) —
//     the forward stays paired with its bwd; if Stage 7.x reintroduces
//     CUDA's optimisations both kernels move together.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE              = 16u;
constant constexpr uint  BLOCK_SIZE             = TILE_SIZE * TILE_SIZE;
constant constexpr float MAX_ALPHA              = 0.99f;
constant constexpr float MIN_ONE_MINUS_ALPHA    = 1e-7f;

kernel void rasterize_to_pixels_3dgs_bwd(
    // gradient outputs (atomic — one per leaf input)
    device atomic_float*        v_means2d        [[buffer(0)]], // [I*N, 2] flat
    device atomic_float*        v_conics         [[buffer(1)]], // [I*N, 3] flat
    device atomic_float*        v_colors         [[buffer(2)]], // [I*N, 3] flat
    device atomic_float*        v_opacities      [[buffer(3)]], // [I*N] flat
    // forward inputs
    device const packed_float2* means2d          [[buffer(4)]],
    device const packed_float3* conics           [[buffer(5)]],
    device const float*         colors           [[buffer(6)]], // [I*N, 3]
    device const float*         opacities        [[buffer(7)]], // [I*N]
    // forward outputs needed for the backward
    device const float*         render_alphas    [[buffer(8)]], // [I, H, W]
    device const int*           last_ids         [[buffer(9)]], // [I, H, W]
    // upstream gradients
    device const float*         v_render_colors  [[buffer(10)]], // [I, H, W, 3]
    device const float*         v_render_alphas  [[buffer(11)]], // [I, H, W]
    // intersections
    device const int*           tile_offsets     [[buffer(12)]], // [I, th, tw]
    device const int*           flatten_ids      [[buffer(13)]], // [n_isects]
    // shape
    constant uint&              image_width      [[buffer(14)]],
    constant uint&              image_height     [[buffer(15)]],
    constant uint&              tile_width       [[buffer(16)]],
    constant uint&              tile_height      [[buffer(17)]],
    constant uint&              n_isects         [[buffer(18)]],
    constant uint&              num_images       [[buffer(19)]],
    uint3                       gid              [[thread_position_in_grid]],
    uint3                       tg_id            [[threadgroup_position_in_grid]]
) {
    constexpr uint CDIM = 3u;

    const uint image_id = gid.z;
    const uint tile_id = tg_id.y * tile_width + tg_id.x;
    const uint pix_x = gid.x;
    const uint pix_y = gid.y;
    const bool inside = (pix_x < image_width) && (pix_y < image_height);
    if (!inside) {
        return;
    }
    const uint pix_id = pix_y * image_width + pix_x;
    const uint pix_base = image_id * image_height * image_width + pix_id;

    const uint tile_offset_idx = image_id * tile_height * tile_width + tile_id;
    int range_start = tile_offsets[tile_offset_idx];
    int range_end;
    bool is_last_tile = (image_id == num_images - 1)
                        && (tile_id == tile_width * tile_height - 1);
    if (is_last_tile) {
        range_end = (int)n_isects;
    } else {
        range_end = tile_offsets[tile_offset_idx + 1];
    }

    // Per-pixel state from the forward.
    const float T_final = 1.0f - render_alphas[pix_base];
    float       T = T_final;
    float       buffer[CDIM] = {0.0f, 0.0f, 0.0f};
    const int   bin_final = last_ids[pix_base];

    // Upstream gradient for this pixel.
    const float v_render_c[CDIM] = {
        v_render_colors[pix_base * CDIM + 0],
        v_render_colors[pix_base * CDIM + 1],
        v_render_colors[pix_base * CDIM + 2],
    };
    const float v_render_a = v_render_alphas[pix_base];

    const float px = (float)pix_x + 0.5f;
    const float py = (float)pix_y + 0.5f;

    // Iterate back-to-front through gaussians that this pixel actually
    // composited (idx <= bin_final). Going from `bin_final` down to
    // `range_start` keeps the relationship T_n = T_{n+1} * 1/(1 - alpha_n)
    // walking from "outermost" to "innermost".
    for (int idx = bin_final; idx >= range_start; --idx) {
        const int g = flatten_ids[idx];
        const float2 xy = means2d[g];
        const float3 cn = conics[g];
        const float opac = opacities[g];

        const float dx = xy.x - px;
        const float dy = xy.y - py;
        const float sigma = 0.5f * (cn.x * dx * dx + cn.z * dy * dy)
                          + cn.y * dx * dy;
        const float vis = exp(-sigma);
        float alpha = min(MAX_ALPHA, opac * vis);
        if (sigma < 0.0f) {
            alpha = 0.0f;
        }

        // Rewind transmittance: T_at_g = T_after_g / (1 - alpha)
        const float ra = 1.0f / max(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
        T *= ra;
        const float fac = alpha * T;

        // Local gradient contributions for this (pixel, gaussian) pair.
        float v_rgb_local[CDIM];
        v_rgb_local[0] = fac * v_render_c[0];
        v_rgb_local[1] = fac * v_render_c[1];
        v_rgb_local[2] = fac * v_render_c[2];

        float v_alpha = 0.0f;
        const uint c_off = (uint)g * CDIM;
        v_alpha += (colors[c_off + 0] * T - buffer[0] * ra) * v_render_c[0];
        v_alpha += (colors[c_off + 1] * T - buffer[1] * ra) * v_render_c[1];
        v_alpha += (colors[c_off + 2] * T - buffer[2] * ra) * v_render_c[2];
        v_alpha += T_final * ra * v_render_a;

        float v_opacity_local = 0.0f;
        float v_xy_local_x = 0.0f, v_xy_local_y = 0.0f;
        float v_conic_local_x = 0.0f, v_conic_local_y = 0.0f, v_conic_local_z = 0.0f;
        if (opac * vis <= MAX_ALPHA) {
            const float v_sigma = -opac * vis * v_alpha;
            v_conic_local_x = 0.5f * v_sigma * dx * dx;
            v_conic_local_y =        v_sigma * dx * dy;
            v_conic_local_z = 0.5f * v_sigma * dy * dy;
            v_xy_local_x = v_sigma * (cn.x * dx + cn.y * dy);
            v_xy_local_y = v_sigma * (cn.y * dx + cn.z * dy);
            v_opacity_local = vis * v_alpha;
        }

        // Update the running back-buffer BEFORE the next iteration.
        buffer[0] += colors[c_off + 0] * fac;
        buffer[1] += colors[c_off + 1] * fac;
        buffer[2] += colors[c_off + 2] * fac;

        // Scatter gradients atomically.
        atomic_fetch_add_explicit(v_colors + c_off + 0, v_rgb_local[0], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 1, v_rgb_local[1], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 2, v_rgb_local[2], memory_order_relaxed);

        atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 0, v_conic_local_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 1, v_conic_local_y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 2, v_conic_local_z, memory_order_relaxed);

        atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 0, v_xy_local_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 1, v_xy_local_y, memory_order_relaxed);

        atomic_fetch_add_explicit(v_opacities + (uint)g, v_opacity_local, memory_order_relaxed);
    }
}
