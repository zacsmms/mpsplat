// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 3DGS projection (forward).
// Port of `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` — pinhole branch with
// `quats + scales` covariance source, no compensations. Other paths
// (covars-given, ortho, fisheye, calc_compensations=true) fall back to the
// torch reference in `_torch_impl._fully_fused_projection`.
//
// Layout: one thread per (batch, camera, gauss). Thread-position-in-grid
// directly indexes the flat `[B*C*N]` workspace.

#include <metal_stdlib>
using namespace metal;

constant constexpr float ALPHA_THRESHOLD  = 1.0f / 255.0f;
constant constexpr float GAUSSIAN_EXTEND  = 3.33f;

inline float3x3 quat_to_rotmat(float4 q) {
    // q = (w, x, y, z), normalize before computing R.
    float inv_norm = rsqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float w = q.x * inv_norm;
    float x = q.y * inv_norm;
    float y = q.z * inv_norm;
    float z = q.w * inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    // Each float3 is one column (column-major, like glm).
    return float3x3(
        float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz),       2.0f * (xz - wy)),
        float3(2.0f * (xy - wz),        1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
        float3(2.0f * (xz + wy),        2.0f * (yz - wx),       1.0f - 2.0f * (x2 + y2))
    );
}

// C = R * S * S * R^T from quaternion + per-axis scale.
inline float3x3 quat_scale_to_covar(float4 q, float3 s) {
    float3x3 R = quat_to_rotmat(q);
    float3x3 M = float3x3(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    return M * transpose(M);
}

kernel void projection_ewa_3dgs_fused_fwd(
    // outputs
    device int*                 radii          [[buffer(0)]],   // [B*C*N, 2]
    device float*               means2d        [[buffer(1)]],   // [B*C*N, 2]
    device float*               depths         [[buffer(2)]],   // [B*C*N]
    device float*               conics         [[buffer(3)]],   // [B*C*N, 3]
    // inputs
    device const packed_float3* means          [[buffer(4)]],   // [B*N, 3]
    device const float*         quats          [[buffer(5)]],   // [B*N, 4]
    device const packed_float3* scales         [[buffer(6)]],   // [B*N, 3]
    device const float*         viewmats       [[buffer(7)]],   // [B*C, 16]
    device const float*         Ks             [[buffer(8)]],   // [B*C, 9]
    device const float*         opacities      [[buffer(9)]],   // [B*N]
    // shape / constants
    constant uint&              B              [[buffer(10)]],
    constant uint&              C              [[buffer(11)]],
    constant uint&              N              [[buffer(12)]],
    constant uint&              image_width    [[buffer(13)]],
    constant uint&              image_height   [[buffer(14)]],
    constant float&             eps2d          [[buffer(15)]],
    constant float&             near_plane     [[buffer(16)]],
    constant float&             far_plane      [[buffer(17)]],
    constant float&             radius_clip    [[buffer(18)]],
    constant uint&              has_opacities  [[buffer(19)]],
    uint                        idx            [[thread_position_in_grid]]
) {
    if (idx >= B * C * N) {
        return;
    }
    const uint bid = idx / (C * N);
    const uint cid = (idx / N) % C;
    const uint gid = idx % N;

    // viewmat is row-major flattened [4, 4]. Build column-major float3x3 R.
    const uint vm_off = (bid * C + cid) * 16u;
    const float v0 = viewmats[vm_off + 0];
    const float v1 = viewmats[vm_off + 1];
    const float v2 = viewmats[vm_off + 2];
    const float v3 = viewmats[vm_off + 3];
    const float v4 = viewmats[vm_off + 4];
    const float v5 = viewmats[vm_off + 5];
    const float v6 = viewmats[vm_off + 6];
    const float v7 = viewmats[vm_off + 7];
    const float v8 = viewmats[vm_off + 8];
    const float v9 = viewmats[vm_off + 9];
    const float v10 = viewmats[vm_off + 10];
    const float v11 = viewmats[vm_off + 11];
    const float3x3 R = float3x3(
        float3(v0, v4, v8),   // 1st column
        float3(v1, v5, v9),   // 2nd column
        float3(v2, v6, v10)   // 3rd column
    );
    const float3 t = float3(v3, v7, v11);

    const uint mn_off = bid * N + gid;
    const float3 mean_w = means[mn_off];
    const float3 mean_c = R * mean_w + t;

    // Always emit depth and the un-projected mean (fx*x/z+cx, fy*y/z+cy)
    // so culled-radii rows still carry the same values as the torch
    // reference. The rasterizer treats `radii == 0` as the cull signal.
    const uint k_off_pre = (bid * C + cid) * 9u;
    const float fx_pre = Ks[k_off_pre + 0];
    const float fy_pre = Ks[k_off_pre + 4];
    const float cx_pre = Ks[k_off_pre + 2];
    const float cy_pre = Ks[k_off_pre + 5];
    const float rz_pre = 1.0f / max(mean_c.z, 1e-12f);
    means2d[idx * 2 + 0] = fx_pre * mean_c.x * rz_pre + cx_pre;
    means2d[idx * 2 + 1] = fy_pre * mean_c.y * rz_pre + cy_pre;
    depths[idx] = mean_c.z;
    radii[idx * 2 + 0] = 0;
    radii[idx * 2 + 1] = 0;
    conics[idx * 3 + 0] = 0.0f;
    conics[idx * 3 + 1] = 0.0f;
    conics[idx * 3 + 2] = 0.0f;

    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        return;
    }

    const float4 q = float4(
        quats[mn_off * 4 + 0],
        quats[mn_off * 4 + 1],
        quats[mn_off * 4 + 2],
        quats[mn_off * 4 + 3]
    );
    const float3 s = scales[mn_off];
    const float3x3 covar_w = quat_scale_to_covar(q, s);
    const float3x3 covar_c = R * covar_w * transpose(R);

    // Pinhole projection — affine approximation around mean_c with FOV
    // clamping (matches `persp_proj` in Utils.cuh).
    const uint k_off = (bid * C + cid) * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];

    const float tan_fovx = 0.5f * (float)image_width  / fx;
    const float tan_fovy = 0.5f * (float)image_height / fy;
    const float lim_x_pos = ((float)image_width  - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = ((float)image_height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float rz = 1.0f / mean_c.z;
    const float rz2 = rz * rz;
    const float tx = mean_c.z * min(lim_x_pos, max(-lim_x_neg, mean_c.x * rz));
    const float ty = mean_c.z * min(lim_y_pos, max(-lim_y_neg, mean_c.y * rz));

    // J is 2x3:  [ fx*rz   0       -fx*tx*rz2 ]
    //           [ 0        fy*rz   -fy*ty*rz2 ]
    const float j00 = fx * rz;
    const float j02 = -fx * tx * rz2;
    const float j11 = fy * rz;
    const float j12 = -fy * ty * rz2;

    // covar2d = J * covar_c * J^T (output 2x2). Float3x3 is column-major,
    // so covar_c[col][row] reads element (row, col).
    const float c_00 = covar_c[0][0];
    const float c_01 = covar_c[1][0];
    const float c_02 = covar_c[2][0];
    const float c_11 = covar_c[1][1];
    const float c_12 = covar_c[2][1];
    const float c_22 = covar_c[2][2];

    // Row 0 of J:    (j00, 0, j02)
    // Row 1 of J:    (0, j11, j12)
    // tmp[r,c] = J[r,:] @ covar_c[:,c]
    const float t00 = j00 * c_00 + j02 * c_02;
    const float t01 = j00 * c_01 + j02 * c_12;
    const float t02 = j00 * c_02 + j02 * c_22;
    const float t10 = j11 * c_01 + j12 * c_02;
    const float t11 = j11 * c_11 + j12 * c_12;
    const float t12 = j11 * c_12 + j12 * c_22;

    // covar2d[r,c] = tmp[r,:] @ J[c,:]
    float c2_00 = t00 * j00 + t02 * j02;
    float c2_01 = t00 *  0  + t01 * j11 + t02 * j12;
    float c2_11 = t10 *  0  + t11 * j11 + t12 * j12;

    // add_blur: bump diagonals by eps2d, recompute determinant.
    c2_00 += eps2d;
    c2_11 += eps2d;
    float det_blur = c2_00 * c2_11 - c2_01 * c2_01;
    if (det_blur <= 0.0f) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    const float inv_det = 1.0f / det_blur;
    const float i00 =  c2_11 * inv_det;
    const float i01 = -c2_01 * inv_det;
    const float i11 =  c2_00 * inv_det;

    const float mean2d_x = fx * mean_c.x * rz + cx;
    const float mean2d_y = fy * mean_c.y * rz + cy;

    float extend = GAUSSIAN_EXTEND;
    if (has_opacities != 0u) {
        float opacity = opacities[mn_off];
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2 + 0] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * log(opacity / ALPHA_THRESHOLD)));
    }

    const float radius_x = ceil(extend * sqrt(c2_00));
    const float radius_y = ceil(extend * sqrt(c2_11));
    // Always write means2d / depths / conics — only `radii==0` is the cull
    // signal. The torch reference does the same, which keeps the bwd's
    // chain-through-conics consistent for culled-but-still-projected gauss.
    means2d[idx * 2 + 0] = mean2d_x;
    means2d[idx * 2 + 1] = mean2d_y;
    depths[idx] = mean_c.z;
    conics[idx * 3 + 0] = i00;
    conics[idx * 3 + 1] = i01;
    conics[idx * 3 + 2] = i11;

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    if (mean2d_x + radius_x <= 0.0f
     || mean2d_x - radius_x >= (float)image_width
     || mean2d_y + radius_y <= 0.0f
     || mean2d_y - radius_y >= (float)image_height) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    radii[idx * 2 + 0] = (int)radius_x;
    radii[idx * 2 + 1] = (int)radius_y;
}
