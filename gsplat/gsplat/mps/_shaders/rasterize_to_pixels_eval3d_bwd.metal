// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3DGUT eval3d backward rasterizer (pinhole or fisheye + global-shutter + CDIM=3).
// Camera-model selector mirrors the forward kernel; see eval3d_fwd.metal.
//
// Walks the same gaussians as the forward in reverse depth order, recomputes
// alpha to match forward, derives per-pixel gradient contributions, and
// atomic-adds them into per-gaussian gradient buffers.
//
// Forward chain recap:
//   d_origin   = ray_o - mw
//   rqT_d      = R_q^T · d_origin
//   gro        = inv_s · rqT_d                    (componentwise inv_s = 1/s)
//   rqT_dir    = R_q^T · ray_d
//   grd_unnorm = inv_s · rqT_dir
//   grd_n      = max(|grd_unnorm|, 1e-12)
//   grd        = grd_unnorm / grd_n
//   gxg        = cross(grd, gro)
//   grayDist   = |gxg|^2
//   alpha_raw  = opac · exp(-0.5 · grayDist)
//   alpha      = min(alpha_raw, MAX_ALPHA)
//   vis        = alpha · T
//
// Backward (when not alpha-clamped):
//   v_grayDist = -0.5 · alpha_raw · v_alpha
//   v_gxg      = 2 · gxg · v_grayDist
//   v_grd      = cross(gro, v_gxg)
//   v_gro      = cross(v_gxg, grd)
//   v_grd_unnorm = (v_grd - (v_grd · grd) · grd) / grd_n
//   v_rqT_dir  = inv_s · v_grd_unnorm
//   v_rqT_d    = inv_s · v_gro
//   v_inv_s    = rqT_dir · v_grd_unnorm  +  rqT_d · v_gro
//   v_s        = -(inv_s · inv_s) · v_inv_s
//   v_R_q     += outer(ray_d, v_rqT_dir) + outer(d_origin, v_rqT_d)
//   v_d_origin = R_q · v_rqT_d
//   v_mw       = -v_d_origin
//   v_q_norm  ← quat_to_rotmat VJP(v_R_q)
//   v_q_raw   ← L2-normalize VJP(v_q_norm, q_norm, q_raw_norm)

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE                = 16u;
constant constexpr float MAX_ALPHA                = 1.0f - 0.0316227766f;  // 1 - sqrt(1e-3)
constant constexpr float MIN_ONE_MINUS_ALPHA      = 1e-7f;

inline float3x3 quat_to_rotmat_norm(float w, float x, float y, float z) {
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(
        float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz),       2.0f * (xz - wy)),
        float3(2.0f * (xy - wz),        1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
        float3(2.0f * (xz + wy),        2.0f * (yz - wx),       1.0f - 2.0f * (x2 + y2))
    );
}

inline float3 pixel_to_ray_d_cam(
    uint   cm_id,
    float  px, float py,
    float  fx, float fy, float cx, float cy,
    device const float* radial_per_cam,
    constant uint& newton_iters
) {
    const float u = (px - cx) / fx;
    const float v = (py - cy) / fy;
    if (cm_id == 2u) {
        const float r = sqrt(u * u + v * v);
        if (r < 1e-12f) {
            return float3(0.0f, 0.0f, 1.0f);
        }
        const float k1 = radial_per_cam[0];
        const float k2 = radial_per_cam[1];
        const float k3 = radial_per_cam[2];
        const float k4 = radial_per_cam[3];
        float theta = r;
        for (uint it = 0; it < newton_iters; ++it) {
            const float t2 = theta * theta;
            const float t3 = theta * t2;
            const float t5 = t3 * t2;
            const float t7 = t5 * t2;
            const float t9 = t7 * t2;
            const float poly = theta + k1 * t3 + k2 * t5 + k3 * t7 + k4 * t9;
            const float dpoly = 1.0f
                + 3.0f * k1 * t2
                + 5.0f * k2 * t2 * t2
                + 7.0f * k3 * t2 * t2 * t2
                + 9.0f * k4 * t2 * t2 * t2 * t2;
            const float resid = poly - r;
            theta = theta - resid / max(dpoly, 1e-12f);
        }
        const float st = sin(theta);
        const float ct = cos(theta);
        const float inv_r = 1.0f / r;
        return float3(st * u * inv_r, st * v * inv_r, ct);
    }
    return float3(u, v, 1.0f);
}

kernel void rasterize_to_pixels_eval3d_bwd(
    // gradient outputs (atomic — shared across cameras for means/quats/scales)
    device atomic_float*        v_means        [[buffer(0)]],   // [B*N, 3]
    device atomic_float*        v_quats        [[buffer(1)]],   // [B*N, 4]
    device atomic_float*        v_scales       [[buffer(2)]],   // [B*N, 3]
    device atomic_float*        v_colors       [[buffer(3)]],   // [I*N, 3]
    device atomic_float*        v_opacities    [[buffer(4)]],   // [I*N]
    // forward inputs
    device const packed_float3* means_w        [[buffer(5)]],   // [B*N, 3]
    device const float*         quats          [[buffer(6)]],   // [B*N, 4]
    device const packed_float3* scales         [[buffer(7)]],   // [B*N, 3]
    device const float*         colors         [[buffer(8)]],   // [I*N, 3]
    device const float*         opacities      [[buffer(9)]],   // [I*N]
    device const float*         viewmats       [[buffer(10)]],  // [I, 16]
    device const float*         Ks             [[buffer(11)]],  // [I, 9]
    // forward outputs needed for the backward
    device const float*         render_alphas  [[buffer(12)]],  // [I, H, W]
    device const int*           last_ids       [[buffer(13)]],  // [I, H, W]
    // upstream grads
    device const float*         v_render_colors[[buffer(14)]],  // [I, H, W, 3]
    device const float*         v_render_alphas[[buffer(15)]],  // [I, H, W]
    // intersections
    device const int*           tile_offsets   [[buffer(16)]],
    device const int*           flatten_ids    [[buffer(17)]],
    // shape / constants
    constant uint&              image_width    [[buffer(18)]],
    constant uint&              image_height   [[buffer(19)]],
    constant uint&              tile_width     [[buffer(20)]],
    constant uint&              tile_height    [[buffer(21)]],
    constant uint&              n_isects       [[buffer(22)]],
    constant uint&              num_images     [[buffer(23)]],
    constant uint&              N              [[buffer(24)]],
    constant uint&              C              [[buffer(25)]],
    constant uint&              cm_id          [[buffer(26)]],
    device const float*         radial_coeffs  [[buffer(27)]],   // [I, 4]
    constant uint&              newton_iters   [[buffer(28)]],
    uint3                       gid            [[thread_position_in_grid]],
    uint3                       tg_id          [[threadgroup_position_in_grid]]
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

    // Reconstruct R_view, t_view, K for this image, then build the camera ray.
    const uint vm_off = image_id * 16u;
    const float v0 = viewmats[vm_off + 0],  v1 = viewmats[vm_off + 1];
    const float v2 = viewmats[vm_off + 2],  v3 = viewmats[vm_off + 3];
    const float v4 = viewmats[vm_off + 4],  v5 = viewmats[vm_off + 5];
    const float v6 = viewmats[vm_off + 6],  v7 = viewmats[vm_off + 7];
    const float v8 = viewmats[vm_off + 8],  v9 = viewmats[vm_off + 9];
    const float v10 = viewmats[vm_off + 10], v11 = viewmats[vm_off + 11];
    const float3x3 R_view = float3x3(
        float3(v0, v4, v8), float3(v1, v5, v9), float3(v2, v6, v10)
    );
    const float3 t_view = float3(v3, v7, v11);
    const float3x3 R_view_T = transpose(R_view);
    const float3 ray_o_world = -(R_view_T * t_view);

    const uint k_off = image_id * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];
    const float px_f = (float)pix_x + 0.5f;
    const float py_f = (float)pix_y + 0.5f;
    device const float* radial_per_cam = radial_coeffs + image_id * 4u;
    float3 ray_d_cam = pixel_to_ray_d_cam(
        cm_id, px_f, py_f, fx, fy, cx, cy, radial_per_cam, newton_iters
    );
    ray_d_cam = normalize(ray_d_cam);
    const float3 ray_d_world = R_view_T * ray_d_cam;

    const uint bid = image_id / C;

    const float T_final = 1.0f - render_alphas[pix_base];
    float       T = T_final;
    float       buffer_color[CDIM] = {0.0f, 0.0f, 0.0f};
    const int   bin_final = last_ids[pix_base];

    const float v_render_c[CDIM] = {
        v_render_colors[pix_base * CDIM + 0],
        v_render_colors[pix_base * CDIM + 1],
        v_render_colors[pix_base * CDIM + 2],
    };
    const float v_render_a = v_render_alphas[pix_base];

    for (int idx = bin_final; idx >= range_start; --idx) {
        const int gflat = flatten_ids[idx];
        const uint g_local = (uint)gflat - image_id * N;
        const uint c_off = (uint)gflat * CDIM;
        const float opac = opacities[gflat];
        const uint mn_off = bid * N + g_local;

        // Recompute forward state.
        const float qw_raw = quats[mn_off * 4 + 0];
        const float qx_raw = quats[mn_off * 4 + 1];
        const float qy_raw = quats[mn_off * 4 + 2];
        const float qz_raw = quats[mn_off * 4 + 3];
        const float qnsq = qw_raw * qw_raw + qx_raw * qx_raw
                         + qy_raw * qy_raw + qz_raw * qz_raw;
        if (qnsq < 1e-30f) {
            continue;
        }
        const float qnorm = sqrt(qnsq);
        const float inv_qn = 1.0f / qnorm;
        const float qw = qw_raw * inv_qn;
        const float qx = qx_raw * inv_qn;
        const float qy = qy_raw * inv_qn;
        const float qz = qz_raw * inv_qn;
        const float3x3 R_q = quat_to_rotmat_norm(qw, qx, qy, qz);
        const float3 s = float3(scales[mn_off]);
        if (s.x <= 0.0f || s.y <= 0.0f || s.z <= 0.0f) {
            continue;
        }
        const float3 inv_s = float3(1.0f / s.x, 1.0f / s.y, 1.0f / s.z);
        const float3 mw = float3(means_w[mn_off]);
        const float3 d_origin = ray_o_world - mw;
        const float3x3 R_q_T = transpose(R_q);
        const float3 rqT_d = R_q_T * d_origin;
        const float3 gro = rqT_d * inv_s;
        const float3 rqT_dir = R_q_T * ray_d_world;
        const float3 grd_unnorm = rqT_dir * inv_s;
        const float grd_n = max(length(grd_unnorm), 1e-12f);
        const float3 grd = grd_unnorm / grd_n;
        const float3 gxg = cross(grd, gro);
        const float grayDist = dot(gxg, gxg);
        const float alpha_raw = opac * exp(-0.5f * grayDist);
        float alpha = min(MAX_ALPHA, alpha_raw);
        if (alpha <= 0.0f) {
            continue;
        }

        // Rewind transmittance to before this gauss.
        const float ra = 1.0f / max(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
        T *= ra;
        const float fac = alpha * T;

        // Per-pixel local color grad (atomic-add scatter).
        float v_rgb_local[CDIM];
        v_rgb_local[0] = fac * v_render_c[0];
        v_rgb_local[1] = fac * v_render_c[1];
        v_rgb_local[2] = fac * v_render_c[2];

        // Aggregate v_alpha at this pixel.
        float v_alpha = 0.0f;
        v_alpha += (colors[c_off + 0] * T - buffer_color[0] * ra) * v_render_c[0];
        v_alpha += (colors[c_off + 1] * T - buffer_color[1] * ra) * v_render_c[1];
        v_alpha += (colors[c_off + 2] * T - buffer_color[2] * ra) * v_render_c[2];
        v_alpha += T_final * ra * v_render_a;

        // Update color back-buffer BEFORE further chain.
        buffer_color[0] += colors[c_off + 0] * fac;
        buffer_color[1] += colors[c_off + 1] * fac;
        buffer_color[2] += colors[c_off + 2] * fac;

        // Scatter v_colors atomically.
        atomic_fetch_add_explicit(v_colors + c_off + 0, v_rgb_local[0], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 1, v_rgb_local[1], memory_order_relaxed);
        atomic_fetch_add_explicit(v_colors + c_off + 2, v_rgb_local[2], memory_order_relaxed);

        // Skip the rest of the chain when alpha was clamped (no grad to opac
        // or the geometry chain on the clamped branch).
        if (alpha_raw > MAX_ALPHA) {
            continue;
        }

        // v_opac and v_grayDist.
        const float vis_response = exp(-0.5f * grayDist);  // = alpha_raw / opac
        const float v_opac = vis_response * v_alpha;
        const float v_grayDist = -0.5f * alpha_raw * v_alpha;

        // v_gxg = 2 * gxg * v_grayDist
        const float3 v_gxg = 2.0f * gxg * v_grayDist;

        // cross-product VJP: c = cross(a, b); v_a = cross(b, v_c); v_b = cross(v_c, a).
        const float3 v_grd = cross(gro, v_gxg);
        const float3 v_gro = cross(v_gxg, grd);

        // L2-normalize VJP (grd = grd_unnorm / grd_n).
        const float dot_vg = dot(v_grd, grd);
        const float3 v_grd_unnorm = (v_grd - dot_vg * grd) / grd_n;

        // grd_unnorm = inv_s * rqT_dir; gro = inv_s * rqT_d
        const float3 v_rqT_dir = inv_s * v_grd_unnorm;
        const float3 v_rqT_d   = inv_s * v_gro;
        const float3 v_inv_s   = rqT_dir * v_grd_unnorm + rqT_d * v_gro;

        // v_s = -inv_s² · v_inv_s (componentwise, since inv_s = 1/s).
        const float3 v_s_local = -inv_s * inv_s * v_inv_s;

        // R_q^T VJPs:
        //   rqT_dir = R_q^T @ ray_d_world  ⇒  v_R_q[j, k] += ray_d_world[j] * v_rqT_dir[k]
        //   rqT_d   = R_q^T @ d_origin    ⇒  v_R_q[j, k] += d_origin[j]    * v_rqT_d[k]
        // (Math: `rqT_dir_i = sum_j R_q[j, i] * ray_d_j` ⇒ d/dR_q[j, k] = δ_ki·ray_d_j.)
        // Storage: v_R_q_storage[col=k][row=j] = v_R_q_math[j, k].
        // Build the math matrix as 3 columns indexed by k.
        const float3 v_R_q_col0 = ray_d_world * v_rqT_dir.x + d_origin * v_rqT_d.x;
        const float3 v_R_q_col1 = ray_d_world * v_rqT_dir.y + d_origin * v_rqT_d.y;
        const float3 v_R_q_col2 = ray_d_world * v_rqT_dir.z + d_origin * v_rqT_d.z;

        // d_origin = ray_o - mw  ⇒  v_mw = -v_d_origin = -(R_q @ v_rqT_d).
        const float3 v_d_origin = R_q * v_rqT_d;
        const float3 v_mw_local = -v_d_origin;

        // quat-to-rotmat VJP: see projection_ewa_3dgs_fused_bwd.metal for derivation.
        const float vR00 = v_R_q_col0[0];
        const float vR10 = v_R_q_col0[1];
        const float vR20 = v_R_q_col0[2];
        const float vR01 = v_R_q_col1[0];
        const float vR11 = v_R_q_col1[1];
        const float vR21 = v_R_q_col1[2];
        const float vR02 = v_R_q_col2[0];
        const float vR12 = v_R_q_col2[1];
        const float vR22 = v_R_q_col2[2];

        const float vq_w = 2.0f * (
              qz * (vR10 - vR01)
            + qy * (vR02 - vR20)
            + qx * (vR21 - vR12)
        );
        const float vq_x = 2.0f * (
              qy * (vR01 + vR10)
            + qz * (vR02 + vR20)
            - 2.0f * qx * (vR11 + vR22)
            + qw * (vR21 - vR12)
        );
        const float vq_y = 2.0f * (
              qx * (vR01 + vR10)
            + qz * (vR12 + vR21)
            - 2.0f * qy * (vR00 + vR22)
            + qw * (vR02 - vR20)
        );
        const float vq_z = 2.0f * (
              qx * (vR02 + vR20)
            + qy * (vR12 + vR21)
            - 2.0f * qz * (vR00 + vR11)
            + qw * (vR10 - vR01)
        );

        // L2-normalize VJP for q.
        const float dot_vqn = vq_w * qw + vq_x * qx + vq_y * qy + vq_z * qz;
        const float vq_w_raw = (vq_w - dot_vqn * qw) * inv_qn;
        const float vq_x_raw = (vq_x - dot_vqn * qx) * inv_qn;
        const float vq_y_raw = (vq_y - dot_vqn * qy) * inv_qn;
        const float vq_z_raw = (vq_z - dot_vqn * qz) * inv_qn;

        // Atomic-add scatter.
        atomic_fetch_add_explicit(v_opacities + gflat, v_opac, memory_order_relaxed);
        atomic_fetch_add_explicit(v_means + mn_off * 3 + 0, v_mw_local.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_means + mn_off * 3 + 1, v_mw_local.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_means + mn_off * 3 + 2, v_mw_local.z, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 0, vq_w_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 1, vq_x_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 2, vq_y_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 3, vq_z_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 0, v_s_local.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 1, v_s_local.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 2, v_s_local.z, memory_order_relaxed);
    }
}
