// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3DGUT eval3d forward rasterizer (pinhole or fisheye + global-shutter).
// camera_model_id selector:
//   0  pinhole, no distortion          (matches `_PerfectPinholeCameraModel`)
//   2  OpenCV-fisheye + radial[4]       (matches `_OpenCVFisheyeCameraModel`)
// Pinhole + radial/tangential/thin-prism (cm_id=1) and ftheta would require
// Newton-iteration undistortion in MSL with non-trivial residual functions;
// those still fall back to the torch reference.
//
// Mirrors `_torch_impl_eval3d._rasterize_to_pixels_eval3d` for the simplest
// path. Each thread owns one pixel; the threadgroup covers a 16×16 tile.
// Per-Gaussian math is world-space ray–Gaussian distance (matches
// `_compute_ray_gaussian_distance`):
//
//   iscl_rot = (R_q · diag(1/s))^T  =  diag(1/s) · R_q^T            (3x3)
//   gro      = iscl_rot · (ray_o - mean_w)                          (3-vec)
//   grd      = normalize(iscl_rot · ray_d)                          (3-vec)
//   sigma    = 0.5 * |cross(grd, gro)|^2                            (scalar)
//   alpha    = clamp(opac · exp(-sigma), 0, MAX_ALPHA)
//
// Camera ray (pinhole, global shutter):
//   ray_d_cam   = normalize((px - cx) / fx, (py - cy) / fy, 1)
//   ray_o_world = -R_view^T · t_view
//   ray_d_world = R_view^T · ray_d_cam
//
// CDIM=3 hard-coded (matches the 3DGS rasterizer kernels). Backgrounds and
// masks are unsupported here; callers fall back to the torch path for those.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE                = 16u;
constant constexpr float MAX_ALPHA                = 1.0f - 0.0316227766f;  // 1 - sqrt(1e-3)
constant constexpr float TRANSMITTANCE_THRESHOLD  = 1e-3f;

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

// Pixel-center → camera-space ray direction (un-normalized; caller normalizes).
//   cm_id=0: pure pinhole.
//   cm_id=2: OpenCV fisheye, 4 radial coeffs. Newton iteration on the
//            forward odd polynomial gives theta s.t.
//            theta + k1 θ³ + k2 θ⁵ + k3 θ⁷ + k4 θ⁹ = ||uv||.
//            Then ray_d = (sin(theta)·uv.x/||uv||, sin(theta)·uv.y/||uv||, cos(theta)).
inline float3 pixel_to_ray_d_cam(
    uint   cm_id,
    float  px, float py,
    float  fx, float fy, float cx, float cy,
    device const float* radial_per_cam,   // 4 floats (only used for fisheye)
    constant uint& newton_iters
) {
    const float u = (px - cx) / fx;
    const float v = (py - cy) / fy;
    if (cm_id == 2u) {
        const float r = sqrt(u * u + v * v);
        if (r < 1e-12f) {
            return float3(0.0f, 0.0f, 1.0f);  // image center
        }
        const float k1 = radial_per_cam[0];
        const float k2 = radial_per_cam[1];
        const float k3 = radial_per_cam[2];
        const float k4 = radial_per_cam[3];
        // Newton-solve for theta. Initial guess: theta ≈ r (small-distortion).
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
    // cm_id == 0: pinhole.
    return float3(u, v, 1.0f);
}

kernel void rasterize_to_pixels_eval3d_fwd(
    device float*               render_colors  [[buffer(0)]],  // [I, H, W, 3]
    device float*               render_alphas  [[buffer(1)]],  // [I, H, W, 1]
    device int*                 last_ids       [[buffer(2)]],  // [I, H, W]
    // shared per-gauss inputs
    device const packed_float3* means_w        [[buffer(3)]],  // [B*N, 3]
    device const float*         quats          [[buffer(4)]],  // [B*N, 4]
    device const packed_float3* scales         [[buffer(5)]],  // [B*N, 3]
    // per-image inputs
    device const float*         colors         [[buffer(6)]],  // [I*N, 3]
    device const float*         opacities      [[buffer(7)]],  // [I*N]
    device const float*         viewmats       [[buffer(8)]],  // [I, 16]
    device const float*         Ks             [[buffer(9)]],  // [I, 9]
    // intersections
    device const int*           tile_offsets   [[buffer(10)]], // [I, tile_h, tile_w]
    device const int*           flatten_ids    [[buffer(11)]], // [n_isects]
    // shape / constants
    constant uint&              image_width    [[buffer(12)]],
    constant uint&              image_height   [[buffer(13)]],
    constant uint&              tile_width     [[buffer(14)]],
    constant uint&              tile_height    [[buffer(15)]],
    constant uint&              n_isects       [[buffer(16)]],
    constant uint&              num_images     [[buffer(17)]],
    constant uint&              N              [[buffer(18)]],
    constant uint&              C              [[buffer(19)]],
    constant uint&              cm_id          [[buffer(20)]],
    device const float*         radial_coeffs  [[buffer(21)]],   // [I, 4]
    constant uint&              newton_iters   [[buffer(22)]],
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

    // Reconstruct R_view, t_view, K for this image.
    // means_w/quats/scales are SHARED across cameras (indexed by [b * N + n]),
    // while flatten_ids encodes (image_id * N + gauss_id) like the 3DGS kernel.
    // image_id = b * C + c, so the batch index is bid = image_id / C.
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
    // R_view^T (camera-to-world rotation).
    const float3x3 R_view_T = transpose(R_view);
    const float3 ray_o_world = -(R_view_T * t_view);

    const uint k_off = image_id * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];

    // Camera ray in world space.
    const float px = (float)pix_x + 0.5f;
    const float py = (float)pix_y + 0.5f;
    device const float* radial_per_cam = radial_coeffs + image_id * 4u;
    float3 ray_d_cam = pixel_to_ray_d_cam(
        cm_id, px, py, fx, fy, cx, cy, radial_per_cam, newton_iters
    );
    ray_d_cam = normalize(ray_d_cam);
    const float3 ray_d_world = R_view_T * ray_d_cam;

    const uint bid = image_id / C;

    float T = 1.0f;
    int   cur_idx = 0;
    bool  done = !inside;
    float pix_color[CDIM] = {0.0f, 0.0f, 0.0f};

    if (!done) {
        for (int idx = range_start; idx < range_end; ++idx) {
            const int gflat = flatten_ids[idx];
            // gflat encodes (image_id * N + gauss_local) per `_isect_tiles`;
            // gauss-local index for this image:
            const uint g_local = (uint)gflat - image_id * N;
            // Per-image color/opacity slot.
            const uint c_off = (uint)gflat * CDIM;
            const float opac = opacities[gflat];
            // Shared (across cameras) gauss params.
            const uint mn_off = bid * N + g_local;
            const float3 mw = float3(means_w[mn_off]);
            const float qw = quats[mn_off * 4 + 0];
            const float qx = quats[mn_off * 4 + 1];
            const float qy = quats[mn_off * 4 + 2];
            const float qz = quats[mn_off * 4 + 3];
            const float qnsq = qw * qw + qx * qx + qy * qy + qz * qz;
            if (qnsq < 1e-30f) {
                continue;
            }
            const float inv_qn = rsqrt(qnsq);
            const float3x3 R_q = quat_to_rotmat_norm(
                qw * inv_qn, qx * inv_qn, qy * inv_qn, qz * inv_qn);
            const float3 s = float3(scales[mn_off]);
            if (s.x <= 0.0f || s.y <= 0.0f || s.z <= 0.0f) {
                continue;
            }
            const float3 inv_s = float3(1.0f / s.x, 1.0f / s.y, 1.0f / s.z);
            // iscl_rot = diag(inv_s) · R_q^T
            // Apply to (ray_o - mw): (R_q^T · (ray_o - mw)) componentwise / s.
            const float3 d_origin = ray_o_world - mw;
            const float3 rqT_d = transpose(R_q) * d_origin;
            const float3 gro = rqT_d * inv_s;
            const float3 rqT_dir = transpose(R_q) * ray_d_world;
            float3 grd_unnorm = rqT_dir * inv_s;
            const float grd_n = max(length(grd_unnorm), 1e-12f);
            const float3 grd = grd_unnorm / grd_n;
            // grayDist = ||cross(grd, gro)||²
            const float3 gxg = cross(grd, gro);
            const float grayDist = dot(gxg, gxg);
            const float power = -0.5f * grayDist;
            float alpha = opac * exp(power);
            alpha = min(MAX_ALPHA, alpha);
            if (alpha <= 0.0f) {
                continue;
            }
            const float next_T = T * (1.0f - alpha);
            if (next_T < TRANSMITTANCE_THRESHOLD) {
                // mark this gauss as last-contributor and bail.
                const float vis = alpha * T;
                pix_color[0] += colors[c_off + 0] * vis;
                pix_color[1] += colors[c_off + 1] * vis;
                pix_color[2] += colors[c_off + 2] * vis;
                cur_idx = idx;
                T = next_T;
                break;
            }
            const float vis = alpha * T;
            pix_color[0] += colors[c_off + 0] * vis;
            pix_color[1] += colors[c_off + 1] * vis;
            pix_color[2] += colors[c_off + 2] * vis;
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
        last_ids[pix_base] = cur_idx;
    }
}
