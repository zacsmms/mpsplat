// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 2DGS projection (forward).
// Port of `gsplat/cuda/csrc/Projection2DGSFused.cu` — pinhole branch, no
// compensations. Outputs:
//   radii [B,C,N,2]     int32  (cull when 0)
//   means2d [B,C,N,2]   float
//   depths [B,C,N]      float
//   ray_transforms [B,C,N,3,3] float (M^T from the paper)
//   normals [B,C,N,3]   float

#include <metal_stdlib>
using namespace metal;

constant constexpr float TWODGS_EXTENT = 3.33f;

inline float3x3 quat_to_rotmat_2dgs(float4 q) {
    float inv_norm = rsqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float w = q.x * inv_norm;
    float x = q.y * inv_norm;
    float y = q.z * inv_norm;
    float z = q.w * inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(
        float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz),       2.0f * (xz - wy)),
        float3(2.0f * (xy - wz),        1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
        float3(2.0f * (xz + wy),        2.0f * (yz - wx),       1.0f - 2.0f * (x2 + y2))
    );
}

kernel void projection_2dgs_fused_fwd(
    // outputs
    device int*                 radii          [[buffer(0)]],   // [B*C*N, 2]
    device float*               means2d        [[buffer(1)]],   // [B*C*N, 2]
    device float*               depths         [[buffer(2)]],   // [B*C*N]
    device float*               ray_transforms [[buffer(3)]],   // [B*C*N, 3, 3]
    device float*               normals        [[buffer(4)]],   // [B*C*N, 3]
    // inputs
    device const packed_float3* means          [[buffer(5)]],   // [B*N, 3]
    device const float*         quats          [[buffer(6)]],   // [B*N, 4]
    device const packed_float3* scales         [[buffer(7)]],   // [B*N, 3]
    device const float*         viewmats       [[buffer(8)]],   // [B*C, 16]
    device const float*         Ks             [[buffer(9)]],   // [B*C, 9]
    // shape / constants
    constant uint&              B              [[buffer(10)]],
    constant uint&              C              [[buffer(11)]],
    constant uint&              N              [[buffer(12)]],
    constant uint&              image_width    [[buffer(13)]],
    constant uint&              image_height   [[buffer(14)]],
    constant float&             near_plane     [[buffer(15)]],
    constant float&             far_plane      [[buffer(16)]],
    constant float&             radius_clip    [[buffer(17)]],
    uint                        idx            [[thread_position_in_grid]]
) {
    if (idx >= B * C * N) {
        return;
    }
    const uint bid = idx / (C * N);
    const uint cid = (idx / N) % C;
    const uint gid = idx % N;

    // Build R, t from row-major viewmat.
    const uint vm_off = (bid * C + cid) * 16u;
    const float v0 = viewmats[vm_off + 0],  v1 = viewmats[vm_off + 1];
    const float v2 = viewmats[vm_off + 2],  v3 = viewmats[vm_off + 3];
    const float v4 = viewmats[vm_off + 4],  v5 = viewmats[vm_off + 5];
    const float v6 = viewmats[vm_off + 6],  v7 = viewmats[vm_off + 7];
    const float v8 = viewmats[vm_off + 8],  v9 = viewmats[vm_off + 9];
    const float v10 = viewmats[vm_off + 10], v11 = viewmats[vm_off + 11];
    const float3x3 R_cw = float3x3(
        float3(v0, v4, v8), float3(v1, v5, v9), float3(v2, v6, v10)
    );
    const float3 t_cw = float3(v3, v7, v11);

    // mean in camera space.
    const uint mn_off = bid * N + gid;
    const float3 mean_w = means[mn_off];
    const float3 mean_c = R_cw * mean_w + t_cw;

    // RS_wl = R_world * diag(scales). For 2DGS, only the first two columns
    // of RS contribute to the ray transform; the third is the normal.
    const float4 q = float4(
        quats[mn_off * 4 + 0],
        quats[mn_off * 4 + 1],
        quats[mn_off * 4 + 2],
        quats[mn_off * 4 + 3]
    );
    const float3 s = scales[mn_off];
    const float3x3 R_wl = quat_to_rotmat_2dgs(q);
    const float3x3 RS_wl = float3x3(R_wl[0] * s.x, R_wl[1] * s.y, R_wl[2] * s.z);
    const float3x3 RS_cl = R_cw * RS_wl;

    // Normal: RS_cl's third column, oriented so that <-normal, mean_c> > 0.
    float3 n_cam = float3(RS_cl[2][0], RS_cl[2][1], RS_cl[2][2]);
    const float dotmc = -dot(n_cam, mean_c);
    if (dotmc < 0.0f) {
        n_cam = -n_cam;
    }

    // T_cl is [..., 3, 3] = [RS_cl[:, :2] | mean_c]. Build it column-major
    // (cols 0,1 from RS_cl; col 2 = mean_c).
    const float3x3 T_cl = float3x3(RS_cl[0], RS_cl[1], mean_c);

    // K is row-major [3,3]; use the upper-left 3x3 directly.
    const uint k_off = (bid * C + cid) * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];
    const float3x3 K3 = float3x3(
        float3(fx, 0.0f, 0.0f),       // col 0
        float3(0.0f, fy, 0.0f),       // col 1
        float3(cx, cy, 1.0f)          // col 2
    );

    const float3x3 T_sl = K3 * T_cl;
    // M = transpose(T_sl) in math. In column-major MSL the storage layout
    // of `M` has M_storage[col][row] = M_math[row, col]. With the transpose
    // we get M_storage[col][row] = T_sl_math[col, row] = T_sl_storage[row][col].
    const float3x3 M = transpose(T_sl);

    // Convenient aliases: in column-major MSL, `M[c]` is column c of the
    // *math* matrix. The torch reference's `M[..., k]` indexes the LAST
    // tensor dim, which is the column. So we need columns here, not rows.
    const float3 col0 = M[0];   // M_math[:, 0]
    const float3 col1 = M[1];   // M_math[:, 1]
    const float3 col2 = M[2];   // M_math[:, 2]

    const float3 test = float3(1.0f, 1.0f, -1.0f);
    const float d_val = dot(col2 * col2, test);
    bool valid = (fabs(d_val) > 0.0f);

    float u = 0.0f, v = 0.0f;
    float ext_x = 0.0f, ext_y = 0.0f;
    if (valid) {
        const float3 f = test / d_val;
        // means2d_x = sum_j  M[j, 0] * M[j, 2] * f[j]  =  dot(col0 * col2, f)
        u  = dot(col0 * col2, f);
        v  = dot(col1 * col2, f);
        const float u2 = dot(col0 * col0, f);
        const float v2 = dot(col1 * col1, f);
        ext_x = sqrt(max(u * u - u2, 1e-4f));
        ext_y = sqrt(max(v * v - v2, 1e-4f));
    }

    const float depth = mean_c.z;
    const bool ok_depth = (depth > near_plane) && (depth < far_plane);

    int rx = (int)ceil(TWODGS_EXTENT * ext_x);
    int ry = (int)ceil(TWODGS_EXTENT * ext_y);
    if (!valid || !ok_depth) {
        rx = 0;
        ry = 0;
    }

    if (rx <= (int)radius_clip && ry <= (int)radius_clip) {
        rx = 0;
        ry = 0;
    }
    if (u + (float)rx <= 0.0f
     || u - (float)rx >= (float)image_width
     || v + (float)ry <= 0.0f
     || v - (float)ry >= (float)image_height) {
        rx = 0;
        ry = 0;
    }

    radii[idx * 2 + 0] = rx;
    radii[idx * 2 + 1] = ry;
    means2d[idx * 2 + 0] = u;
    means2d[idx * 2 + 1] = v;
    depths[idx] = depth;

    // The torch reference transposes once before the AABB math then again
    // before returning, so the actually-returned matrix is `T_sl` (NOT M).
    // In math indexing, T_sl[r, c] is stored at `ray_transforms[r*3 + c]`.
    // T_sl_metal_storage[c][r] = T_sl_math[r, c].
    for (uint r = 0; r < 3; ++r) {
        ray_transforms[idx * 9 + r * 3 + 0] = T_sl[0][r];
        ray_transforms[idx * 9 + r * 3 + 1] = T_sl[1][r];
        ray_transforms[idx * 9 + r * 3 + 2] = T_sl[2][r];
    }

    normals[idx * 3 + 0] = n_cam.x;
    normals[idx * 3 + 1] = n_cam.y;
    normals[idx * 3 + 2] = n_cam.z;
}
