// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 2DGS backward tile rasterizer.
//
// Mirrors `rasterize_to_pixels_3dgs_bwd.metal` but with:
//   * separate `n_buffer` for the rendered-normal channel (3 floats per pixel),
//   * sigma chosen as `0.5 * min(sigma_3d, sigma_2d)`. The selected branch
//     dictates whether the gradient flows into `means2d` (2D branch) or
//     `ray_transforms` (3D branch via the cross-product chain).
//
// Mathematical recap (per gaussian per pixel):
//   M0/M1/M2 := rows of T_sl (ray_transforms[r*3+c] = T_sl[r,c])
//   h_u   := -M0 + px*M2
//   h_v   := -M1 + py*M2
//   tmp   := cross(h_u, h_v)
//   denom := tmp.z if |tmp.z|>eps else 1
//   u, v  := tmp.x/denom, tmp.y/denom
//   sigma_3d := u^2 + v^2
//   sigma_2d := 2*(dx^2 + dy^2)   where dx = mu.x - px, dy = mu.y - py
//   sigma    := 0.5 * min(sigma_3d, sigma_2d)
//
// Branch 2D backward:
//   d sigma / d mu.x = 2*dx ;  d sigma / d mu.y = 2*dy
//
// Branch 3D backward (denom = tmp.z, the common case):
//   d sigma / d tmp.x = u/tmp.z  ; d sigma / d tmp.y = v/tmp.z
//   d sigma / d tmp.z = -(u^2+v^2)/tmp.z = -2*sigma/tmp.z
//   v_h_u = cross(h_v, v_tmp)    ; v_h_v = cross(v_tmp, h_u)
//   v_M0 = -v_h_u                ; v_M1 = -v_h_v
//   v_M2 =  px*v_h_u + py*v_h_v
//
// Edge case (|tmp.z|<=eps): denom is treated as 1, so d sigma / d tmp.z = 0
// and the chain reduces to v_tmp.x = u, v_tmp.y = v, v_tmp.z = 0.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE              = 16u;
constant constexpr uint  BLOCK_SIZE             = TILE_SIZE * TILE_SIZE;
constant constexpr float MAX_ALPHA              = 0.99f;
constant constexpr float MIN_ONE_MINUS_ALPHA    = 1e-7f;

kernel void rasterize_to_pixels_2dgs_bwd(
    // gradient outputs (atomic — one per leaf input)
    device atomic_float*        v_means2d        [[buffer(0)]],  // [I*N, 2]
    device atomic_float*        v_ray_transforms [[buffer(1)]],  // [I*N, 9]
    device atomic_float*        v_colors         [[buffer(2)]],  // [I*N, 3]
    device atomic_float*        v_opacities      [[buffer(3)]],  // [I*N]
    device atomic_float*        v_normals        [[buffer(4)]],  // [I*N, 3]
    // forward inputs
    device const packed_float2* means2d          [[buffer(5)]],
    device const float*         ray_transforms   [[buffer(6)]],
    device const float*         colors           [[buffer(7)]],
    device const float*         opacities        [[buffer(8)]],
    device const packed_float3* normals          [[buffer(9)]],
    // forward outputs needed for the backward
    device const float*         render_alphas    [[buffer(10)]], // [I, H, W]
    device const int*           last_ids         [[buffer(11)]], // [I, H, W]
    // upstream gradients
    device const float*         v_render_colors  [[buffer(12)]], // [I, H, W, 3]
    device const float*         v_render_alphas  [[buffer(13)]], // [I, H, W]
    device const float*         v_render_normals [[buffer(14)]], // [I, H, W, 3]
    // intersections
    device const int*           tile_offsets     [[buffer(15)]],
    device const int*           flatten_ids      [[buffer(16)]],
    // shape
    constant uint&              image_width      [[buffer(17)]],
    constant uint&              image_height     [[buffer(18)]],
    constant uint&              tile_width       [[buffer(19)]],
    constant uint&              tile_height      [[buffer(20)]],
    constant uint&              n_isects         [[buffer(21)]],
    constant uint&              num_images       [[buffer(22)]],
    uint3                       gid              [[thread_position_in_grid]],
    uint3                       tg_id            [[threadgroup_position_in_grid]]
) {
    constexpr uint CDIM = 3u;

    const uint image_id = gid.z;
    const uint tile_id  = tg_id.y * tile_width + tg_id.x;
    const uint pix_x    = gid.x;
    const uint pix_y    = gid.y;
    const bool inside   = (pix_x < image_width) && (pix_y < image_height);
    if (!inside) {
        return;
    }
    const uint pix_id   = pix_y * image_width + pix_x;
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

    const float T_final = 1.0f - render_alphas[pix_base];
    float       T = T_final;
    float       buffer_color[CDIM]  = {0.0f, 0.0f, 0.0f};
    float       buffer_normal[3]    = {0.0f, 0.0f, 0.0f};
    const int   bin_final = last_ids[pix_base];

    const float v_render_c[CDIM] = {
        v_render_colors[pix_base * CDIM + 0],
        v_render_colors[pix_base * CDIM + 1],
        v_render_colors[pix_base * CDIM + 2],
    };
    const float v_render_n[3] = {
        v_render_normals[pix_base * 3 + 0],
        v_render_normals[pix_base * 3 + 1],
        v_render_normals[pix_base * 3 + 2],
    };
    const float v_render_a = v_render_alphas[pix_base];

    const float px = (float)pix_x + 0.5f;
    const float py = (float)pix_y + 0.5f;

    for (int idx = bin_final; idx >= range_start; --idx) {
        const int g = flatten_ids[idx];

        // Recompute forward to determine branch + alpha.
        const uint rt_off = (uint)g * 9u;
        const float3 M0 = float3(
            ray_transforms[rt_off + 0],
            ray_transforms[rt_off + 1],
            ray_transforms[rt_off + 2]);
        const float3 M1 = float3(
            ray_transforms[rt_off + 3],
            ray_transforms[rt_off + 4],
            ray_transforms[rt_off + 5]);
        const float3 M2 = float3(
            ray_transforms[rt_off + 6],
            ray_transforms[rt_off + 7],
            ray_transforms[rt_off + 8]);

        const float3 h_u = -M0 + px * M2;
        const float3 h_v = -M1 + py * M2;
        const float3 tmp = cross(h_u, h_v);

        float u_proj, v_proj, denom;
        bool denom_ok = (fabs(tmp.z) > 1e-12f);
        if (denom_ok) {
            denom = tmp.z;
            u_proj = tmp.x / tmp.z;
            v_proj = tmp.y / tmp.z;
        } else {
            denom = 1.0f;
            u_proj = tmp.x;
            v_proj = tmp.y;
        }
        const float sigma_3d = u_proj * u_proj + v_proj * v_proj;

        const float2 mu = means2d[g];
        const float dx = mu.x - px;
        const float dy = mu.y - py;
        const float sigma_2d = 2.0f * (dx * dx + dy * dy);

        const bool branch_3d = (sigma_3d < sigma_2d);
        const float sigma = 0.5f * (branch_3d ? sigma_3d : sigma_2d);
        const float opac = opacities[g];
        const float vis = exp(-sigma);
        float alpha = min(MAX_ALPHA, opac * vis);
        if (sigma < 0.0f) {
            alpha = 0.0f;
        }

        // Rewind transmittance to before this gauss.
        const float ra = 1.0f / max(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
        T *= ra;
        const float fac = alpha * T;

        // Local visibility-weighted color/normal grads.
        float v_rgb_local[CDIM];
        v_rgb_local[0] = fac * v_render_c[0];
        v_rgb_local[1] = fac * v_render_c[1];
        v_rgb_local[2] = fac * v_render_c[2];
        float v_normal_local[3];
        v_normal_local[0] = fac * v_render_n[0];
        v_normal_local[1] = fac * v_render_n[1];
        v_normal_local[2] = fac * v_render_n[2];

        // Aggregate v_alpha.
        const uint c_off = (uint)g * CDIM;
        float v_alpha = 0.0f;
        v_alpha += (colors[c_off + 0] * T - buffer_color[0] * ra) * v_render_c[0];
        v_alpha += (colors[c_off + 1] * T - buffer_color[1] * ra) * v_render_c[1];
        v_alpha += (colors[c_off + 2] * T - buffer_color[2] * ra) * v_render_c[2];
        const float3 n_g = float3(normals[g]);
        v_alpha += (n_g.x * T - buffer_normal[0] * ra) * v_render_n[0];
        v_alpha += (n_g.y * T - buffer_normal[1] * ra) * v_render_n[1];
        v_alpha += (n_g.z * T - buffer_normal[2] * ra) * v_render_n[2];
        v_alpha += T_final * ra * v_render_a;

        // Propagate v_alpha → v_sigma, v_opac, then sigma → (M, mu).
        float v_opacity_local = 0.0f;
        float v_xy_local_x = 0.0f, v_xy_local_y = 0.0f;
        float v_M0_x = 0.0f, v_M0_y = 0.0f, v_M0_z = 0.0f;
        float v_M1_x = 0.0f, v_M1_y = 0.0f, v_M1_z = 0.0f;
        float v_M2_x = 0.0f, v_M2_y = 0.0f, v_M2_z = 0.0f;

        if (opac * vis <= MAX_ALPHA) {
            const float v_sigma = -opac * vis * v_alpha;
            v_opacity_local = vis * v_alpha;

            if (branch_3d) {
                // sigma = 0.5 * sigma_3d → v_sigma_3d = 0.5 * v_sigma
                const float v_sigma_3d = 0.5f * v_sigma;
                // sigma_3d = u^2 + v^2 → v_u = 2u*v_sigma_3d, v_v = 2v*v_sigma_3d
                const float v_u = 2.0f * u_proj * v_sigma_3d;
                const float v_v = 2.0f * v_proj * v_sigma_3d;
                // u = tmp.x/denom, v = tmp.y/denom
                float v_tmp_x, v_tmp_y, v_tmp_z;
                if (denom_ok) {
                    const float inv_denom = 1.0f / denom;
                    v_tmp_x = v_u * inv_denom;
                    v_tmp_y = v_v * inv_denom;
                    v_tmp_z = -(u_proj * inv_denom) * v_u
                              -(v_proj * inv_denom) * v_v;
                } else {
                    v_tmp_x = v_u;
                    v_tmp_y = v_v;
                    v_tmp_z = 0.0f;
                }
                const float3 v_tmp = float3(v_tmp_x, v_tmp_y, v_tmp_z);
                // tmp = cross(h_u, h_v)
                //   v_h_u = cross(h_v, v_tmp), v_h_v = cross(v_tmp, h_u)
                const float3 v_h_u = cross(h_v, v_tmp);
                const float3 v_h_v = cross(v_tmp, h_u);
                // h_u = -M0 + px*M2 ; h_v = -M1 + py*M2
                v_M0_x = -v_h_u.x; v_M0_y = -v_h_u.y; v_M0_z = -v_h_u.z;
                v_M1_x = -v_h_v.x; v_M1_y = -v_h_v.y; v_M1_z = -v_h_v.z;
                v_M2_x = px * v_h_u.x + py * v_h_v.x;
                v_M2_y = px * v_h_u.y + py * v_h_v.y;
                v_M2_z = px * v_h_u.z + py * v_h_v.z;
            } else {
                // sigma = 0.5 * sigma_2d = dx^2 + dy^2  (dx = mu.x - px)
                v_xy_local_x = 2.0f * dx * v_sigma;
                v_xy_local_y = 2.0f * dy * v_sigma;
            }
        }

        // Update running back-buffers BEFORE the next iteration.
        buffer_color[0] += colors[c_off + 0] * fac;
        buffer_color[1] += colors[c_off + 1] * fac;
        buffer_color[2] += colors[c_off + 2] * fac;
        buffer_normal[0] += n_g.x * fac;
        buffer_normal[1] += n_g.y * fac;
        buffer_normal[2] += n_g.z * fac;

        // Scatter gradients atomically.
        atomic_fetch_add_explicit(v_colors + c_off + 0, v_rgb_local[0], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 1, v_rgb_local[1], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 2, v_rgb_local[2], memory_order_relaxed);

        atomic_fetch_add_explicit(v_normals + (uint)g * 3 + 0, v_normal_local[0], memory_order_relaxed);
        atomic_fetch_add_explicit(v_normals + (uint)g * 3 + 1, v_normal_local[1], memory_order_relaxed);
        atomic_fetch_add_explicit(v_normals + (uint)g * 3 + 2, v_normal_local[2], memory_order_relaxed);

        atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 0, v_xy_local_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 1, v_xy_local_y, memory_order_relaxed);

        atomic_fetch_add_explicit(v_opacities + (uint)g, v_opacity_local, memory_order_relaxed);

        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 0, v_M0_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 1, v_M0_y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 2, v_M0_z, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 3, v_M1_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 4, v_M1_y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 5, v_M1_z, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 6, v_M2_x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 7, v_M2_y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_ray_transforms + rt_off + 8, v_M2_z, memory_order_relaxed);
    }
}
