// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 2DGS forward tile rasterizer.
//
// Mirrors `rasterize_to_pixels_3dgs_fwd.metal` (Stage 6) but uses the
// 2DGS ray-splat alpha formulation from `_torch_impl_2dgs.accumulate_2dgs`.
// Returns colors (CDIM=3 hard-coded), alphas, normals, and last_ids.
//
// Per-pixel math (`ray_transforms` stores T_sl in math row-major; see the
// "2DGS double-transpose dance" gotcha in the plan file):
//
//     px, py     := pixel center
//     M0/M1/M2   := rows 0/1/2 of T_sl (so ray_transforms[r*3 + c] for c=0..2)
//     h_u        := -M0 + px * M2
//     h_v        := -M1 + py * M2
//     tmp        := cross(h_u, h_v)
//     u, v       := tmp.x / tmp.z, tmp.y / tmp.z
//     sigma_3d   := u*u + v*v
//     sigma_2d   := 2 * |xy - mu|^2
//     sigma      := 0.5 * min(sigma_3d, sigma_2d)
//     alpha      := clamp(opac * exp(-sigma), 0, MAX_ALPHA)
//
// Distortion / median-depth are out of scope for now (returned as zeros by
// the host wrapper) — only used by 2DGS surface losses, deferred.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE                = 16u;
constant constexpr uint  BLOCK_SIZE               = TILE_SIZE * TILE_SIZE;
constant constexpr float MAX_ALPHA                = 0.99f;
constant constexpr float ALPHA_THRESHOLD          = 1.0f / 255.0f;
constant constexpr float TRANSMITTANCE_THRESHOLD  = 1e-4f;

// `packed_float3` is 12 bytes (3 contiguous floats) and matches a torch
// tensor's row stride for [..., 3] layouts. `float3` is 16-byte aligned and
// would mis-read every other element of a torch buffer — see the
// `packed_float3 vs float3` gotcha in the plan file.
kernel void rasterize_to_pixels_2dgs_fwd(
    device float*               render_colors  [[buffer(0)]],  // [I, H, W, 3]
    device float*               render_alphas  [[buffer(1)]],  // [I, H, W, 1]
    device float*               render_normals [[buffer(2)]],  // [I, H, W, 3]
    device int*                 last_ids       [[buffer(3)]],  // [I, H, W]
    device const packed_float2* means2d        [[buffer(4)]],  // [I*N, 2]
    device const float*         ray_transforms [[buffer(5)]],  // [I*N, 3, 3] row-major (T_sl_math)
    device const float*         colors         [[buffer(6)]],  // [I*N, 3]
    device const float*         opacities      [[buffer(7)]],  // [I*N]
    device const packed_float3* normals        [[buffer(8)]],  // [I*N, 3]
    device const int*           tile_offsets   [[buffer(9)]],  // [I, tile_h, tile_w]
    device const int*           flatten_ids    [[buffer(10)]], // [n_isects]
    constant uint&              image_width    [[buffer(11)]],
    constant uint&              image_height   [[buffer(12)]],
    constant uint&              tile_width     [[buffer(13)]],
    constant uint&              tile_height    [[buffer(14)]],
    constant uint&              n_isects       [[buffer(15)]],
    constant uint&              num_images     [[buffer(16)]],
    uint3                       gid            [[thread_position_in_grid]],
    uint3                       lid            [[thread_position_in_threadgroup]],
    uint3                       tg_id          [[threadgroup_position_in_grid]]
) {
    constexpr uint CDIM = 3u;

    const uint image_id = gid.z;
    const uint tile_x   = tg_id.x;
    const uint tile_y   = tg_id.y;
    const uint pix_x    = gid.x;
    const uint pix_y    = gid.y;
    const uint tile_id  = tile_y * tile_width + tile_x;
    const uint pix_id   = pix_y * image_width + pix_x;
    const uint tile_offset_idx = image_id * tile_height * tile_width + tile_id;

    const bool inside = (pix_x < image_width) && (pix_y < image_height);

    int range_start = tile_offsets[tile_offset_idx];
    int range_end;
    bool is_last_tile = (image_id == num_images - 1)
                        && (tile_id == tile_width * tile_height - 1);
    if (is_last_tile) {
        range_end = (int)n_isects;
    } else {
        range_end = tile_offsets[tile_offset_idx + 1];
    }

    const float px = (float)pix_x + 0.5f;
    const float py = (float)pix_y + 0.5f;

    float T = 1.0f;
    int   cur_idx = 0;
    bool  done = !inside;
    float pix_color[CDIM]  = {0.0f, 0.0f, 0.0f};
    float pix_normal[3]    = {0.0f, 0.0f, 0.0f};

    if (!done) {
        for (int idx = range_start; idx < range_end; ++idx) {
            const int g = flatten_ids[idx];

            // Read T_sl rows (row-major torch storage: rt[r*3 + c] = T_sl[r, c]).
            const uint rt_off = (uint)g * 9u;
            const float3 M0 = float3(
                ray_transforms[rt_off + 0],
                ray_transforms[rt_off + 1],
                ray_transforms[rt_off + 2]
            );
            const float3 M1 = float3(
                ray_transforms[rt_off + 3],
                ray_transforms[rt_off + 4],
                ray_transforms[rt_off + 5]
            );
            const float3 M2 = float3(
                ray_transforms[rt_off + 6],
                ray_transforms[rt_off + 7],
                ray_transforms[rt_off + 8]
            );

            const float3 h_u = -M0 + px * M2;
            const float3 h_v = -M1 + py * M2;
            const float3 tmp = cross(h_u, h_v);

            float sigma_3d;
            if (fabs(tmp.z) > 1e-12f) {
                const float u = tmp.x / tmp.z;
                const float v = tmp.y / tmp.z;
                sigma_3d = u * u + v * v;
            } else {
                // torch ref divides by 1.0 here, leaving sigma_3d = tmp.x^2 + tmp.y^2.
                // We mirror that: `denom = where(|tmp.z|>1e-12, tmp.z, 1)`.
                sigma_3d = tmp.x * tmp.x + tmp.y * tmp.y;
            }

            const float2 mu = means2d[g];
            const float dx = px - mu.x;
            const float dy = py - mu.y;
            const float sigma_2d = 2.0f * (dx * dx + dy * dy);

            const float sigma = 0.5f * min(sigma_3d, sigma_2d);
            const float opac  = opacities[g];
            float alpha = min(MAX_ALPHA, opac * exp(-sigma));
            if (sigma < 0.0f) {
                alpha = 0.0f;
            }

            // Match torch ref's `(1 - alpha).clamp(min=1e-7)` semantics.
            const float next_T = T * max(1.0f - alpha, 1e-7f);
            const float vis = alpha * T;
            const uint c_off = (uint)g * CDIM;
            pix_color[0] += colors[c_off + 0] * vis;
            pix_color[1] += colors[c_off + 1] * vis;
            pix_color[2] += colors[c_off + 2] * vis;

            const float3 n = float3(normals[g]);
            pix_normal[0] += n.x * vis;
            pix_normal[1] += n.y * vis;
            pix_normal[2] += n.z * vis;

            cur_idx = idx;
            T = next_T;
        }
    }

    if (inside) {
        const uint pix_base = image_id * image_height * image_width + pix_id;
        render_alphas[pix_base] = 1.0f - T;
        render_colors[pix_base * CDIM + 0] = pix_color[0];
        render_colors[pix_base * CDIM + 1] = pix_color[1];
        render_colors[pix_base * CDIM + 2] = pix_color[2];
        render_normals[pix_base * 3 + 0] = pix_normal[0];
        render_normals[pix_base * 3 + 1] = pix_normal[1];
        render_normals[pix_base * 3 + 2] = pix_normal[2];
        last_ids[pix_base] = cur_idx;
    }
}
