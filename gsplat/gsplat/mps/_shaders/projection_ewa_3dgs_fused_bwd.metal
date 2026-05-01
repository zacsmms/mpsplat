// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 3DGS projection (backward).
//
// Reverse-mode chain through the forward in `projection_ewa_3dgs_fused_fwd.metal`:
//   v_conics  → v_covar2d_blur  (-S @ V @ S, sym 2x2)
//   v_covar2d → v_covar_c (J^T @ V @ J)  +  v_J  (2 * V @ J @ covar_c)
//   v_J → v_mean_c (via tx/ty FOV clamp)
//   v_means2d / v_depths → v_mean_c
//   v_mean_c → v_means_w  (R_view^T)
//   v_covar_c → v_covar_w  (R_view^T @ V @ R_view)
//   v_covar_w → v_M  (2 * V @ M)        where M = R_q · diag(s)
//   v_M → v_R_q (per-column scaled by s[j])  +  v_scales[k] = sum_i v_M[i,k] * R_q[i,k]
//   v_R_q → v_q_norm (quat-to-rotmat VJP, Hamilton (w,x,y,z))
//   v_q_norm → v_q   (L2-normalize VJP)
//
// One thread per (b, c, n). v_means / v_quats / v_scales are [B*N, ...] —
// atomic-add scatter so contributions from each camera sum correctly.
//
// Symmetric-matrix convention: every 2x2 / 3x3 sym matrix is stored full
// (off-diagonals mirrored). The chain `J^T V J`, `R^T V R`, `2 V M` reproduces
// the same final v_quats / v_scales / v_means as PyTorch autograd in our
// validation tests; see the derivation comment in the bwd Python wrapper.

#include <metal_stdlib>
using namespace metal;

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

kernel void projection_ewa_3dgs_fused_bwd(
    // grad outputs (atomic — summed over cameras)
    device atomic_float*        v_means        [[buffer(0)]],   // [B*N, 3]
    device atomic_float*        v_quats        [[buffer(1)]],   // [B*N, 4]
    device atomic_float*        v_scales       [[buffer(2)]],   // [B*N, 3]
    // forward inputs
    device const packed_float3* means          [[buffer(3)]],   // [B*N, 3]
    device const float*         quats          [[buffer(4)]],   // [B*N, 4]
    device const packed_float3* scales         [[buffer(5)]],   // [B*N, 3]
    device const float*         viewmats       [[buffer(6)]],   // [B*C, 16]
    device const float*         Ks             [[buffer(7)]],   // [B*C, 9]
    // forward outputs we re-use
    device const int*           radii_in       [[buffer(8)]],   // [B*C*N, 2]
    device const float*         conics_in      [[buffer(9)]],   // [B*C*N, 3]
    // upstream grads
    device const float*         v_means2d      [[buffer(10)]],  // [B*C*N, 2]
    device const float*         v_depths       [[buffer(11)]],  // [B*C*N]
    device const float*         v_conics       [[buffer(12)]],  // [B*C*N, 3]
    // shape
    constant uint&              B              [[buffer(13)]],
    constant uint&              C              [[buffer(14)]],
    constant uint&              N              [[buffer(15)]],
    constant uint&              image_width    [[buffer(16)]],
    constant uint&              image_height   [[buffer(17)]],
    uint                        idx            [[thread_position_in_grid]]
) {
    if (idx >= B * C * N) {
        return;
    }
    const uint bid = idx / (C * N);
    const uint cid = (idx / N) % C;
    const uint gid = idx % N;
    const uint mn_off = bid * N + gid;

    // Reconstruct R_view, t_view, mean_c.
    const uint vm_off = (bid * C + cid) * 16u;
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

    const float3 mean_w = float3(means[mn_off]);
    const float3 mean_c = R_view * mean_w + t_view;

    const uint k_off = (bid * C + cid) * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];

    // Always-emit means2d/depths grad chain. Forward uses
    // `max(mean_c.z, 1e-12)`; gradient through `max()` is 0 at the boundary.
    const float vm2d_x = v_means2d[idx * 2 + 0];
    const float vm2d_y = v_means2d[idx * 2 + 1];
    const float v_d   = v_depths[idx];

    float3 v_mean_c = float3(0.0f, 0.0f, 0.0f);
    {
        const float mz = mean_c.z;
        const float mz_safe = max(mz, 1e-12f);
        const float rzs = 1.0f / mz_safe;
        const float rzs2 = rzs * rzs;
        // dM2d.x/dmean_c.x = fx*rzs;  dM2d.x/dmean_c.z = -fx*mean_c.x*rzs^2
        v_mean_c.x += vm2d_x * fx * rzs;
        v_mean_c.y += vm2d_y * fy * rzs;
        // Only feed the d/dz part if the max() didn't clamp.
        if (mz > 1e-12f) {
            v_mean_c.z += -vm2d_x * fx * mean_c.x * rzs2
                          -vm2d_y * fy * mean_c.y * rzs2;
        }
    }
    v_mean_c.z += v_d;

    // Always run the conics-chain. For depth-culled gauss the forward emits
    // conics=(0,0,0); the -conics V conics formula then produces zero
    // v_covar2d, and the chain naturally contributes 0 to v_M / v_quats /
    // v_scales. The `1/mean_c.z` used by the projection Jacobian is
    // safety-clamped via mz_safe to avoid Inf/NaN for those gauss.
    {
        // Reconstruct R_q, M, covar_w, covar_c, J for the conic-chain.
        const float qw_raw = quats[mn_off * 4 + 0];
        const float qx_raw = quats[mn_off * 4 + 1];
        const float qy_raw = quats[mn_off * 4 + 2];
        const float qz_raw = quats[mn_off * 4 + 3];
        const float qnsq = qw_raw * qw_raw + qx_raw * qx_raw
                         + qy_raw * qy_raw + qz_raw * qz_raw;
        const float qnorm = sqrt(max(qnsq, 1e-24f));
        const float inv_qn = 1.0f / qnorm;
        const float qw = qw_raw * inv_qn;
        const float qx = qx_raw * inv_qn;
        const float qy = qy_raw * inv_qn;
        const float qz = qz_raw * inv_qn;
        const float3x3 R_q = quat_to_rotmat_norm(qw, qx, qy, qz);
        const float3 s = float3(scales[mn_off]);
        const float3x3 M = float3x3(R_q[0] * s.x, R_q[1] * s.y, R_q[2] * s.z);
        const float3x3 covar_w = M * transpose(M);
        const float3x3 covar_c = R_view * covar_w * transpose(R_view);

        // Reconstruct J with FOV clamp.
        const float tan_fovx = 0.5f * (float)image_width  / fx;
        const float tan_fovy = 0.5f * (float)image_height / fy;
        const float lim_x_pos = ((float)image_width  - cx) / fx + 0.3f * tan_fovx;
        const float lim_x_neg =                       cx  / fx + 0.3f * tan_fovx;
        const float lim_y_pos = ((float)image_height - cy) / fy + 0.3f * tan_fovy;
        const float lim_y_neg =                       cy  / fy + 0.3f * tan_fovy;
        // Safety: depth-culled gauss can have mean_c.z very small or negative;
        // matches `max(mean_c.z, 1e-12)` used by the forward for the
        // always-emit means2d. Their conics is 0 so chain still gives 0.
        const float mz_safe_j = max(mean_c.z, 1e-12f);
        const float rz = 1.0f / mz_safe_j;
        const float rz2 = rz * rz;
        const float xz_raw = mean_c.x * rz;
        const float yz_raw = mean_c.y * rz;
        const bool clamp_x_hi = (xz_raw >  lim_x_pos);
        const bool clamp_x_lo = (xz_raw < -lim_x_neg);
        const bool clamp_y_hi = (yz_raw >  lim_y_pos);
        const bool clamp_y_lo = (yz_raw < -lim_y_neg);
        const float xz_c = clamp_x_hi ?  lim_x_pos : (clamp_x_lo ? -lim_x_neg : xz_raw);
        const float yz_c = clamp_y_hi ?  lim_y_pos : (clamp_y_lo ? -lim_y_neg : yz_raw);
        const float tx = mean_c.z * xz_c;
        const float ty = mean_c.z * yz_c;

        const float j00 = fx * rz;
        const float j02 = -fx * tx * rz2;
        const float j11 = fy * rz;
        const float j12 = -fy * ty * rz2;

        // Step 1: v_conics → v_covar2d (sym, off-diag = unique grad / 2 for
        // matrix-multiply chaining; final v_M reuses 2*V which corrects).
        const float vi00 = v_conics[idx * 3 + 0];
        const float vi01 = v_conics[idx * 3 + 1];
        const float vi11 = v_conics[idx * 3 + 2];
        const float s00 = conics_in[idx * 3 + 0];
        const float s01 = conics_in[idx * 3 + 1];
        const float s11 = conics_in[idx * 3 + 2];
        const float v2_00 = -(vi00 * s00 * s00 + vi01 * s00 * s01 + vi11 * s01 * s01);
        const float v2_01 = -(vi00 * s00 * s01
                            + vi01 * 0.5f * (s00 * s11 + s01 * s01)
                            + vi11 * s01 * s11);
        const float v2_11 = -(vi00 * s01 * s01 + vi01 * s01 * s11 + vi11 * s11 * s11);

        // Step 2: v_covar_c = J^T @ V_2d @ J  (3x3 sym).
        // J^T (3x2) rows: (j00, 0), (0, j11), (j02, j12).
        // V_2d @ J (2x3):
        //   row 0: (V[0,0]*j00, V[0,1]*j11, V[0,0]*j02 + V[0,1]*j12)
        //   row 1: (V[0,1]*j00, V[1,1]*j11, V[0,1]*j02 + V[1,1]*j12)
        const float a00 = v2_00 * j00;
        const float a01 = v2_01 * j11;
        const float a02 = v2_00 * j02 + v2_01 * j12;
        const float a10 = v2_01 * j00;
        const float a11 = v2_11 * j11;
        const float a12 = v2_01 * j02 + v2_11 * j12;
        // v_covar_c[i,j] = J^T[i,:] @ (V@J)[:,j]
        const float vcc_00 = j00 * a00;
        const float vcc_01 = j00 * a01;
        const float vcc_02 = j00 * a02;
        const float vcc_10 = j11 * a10;       // == vcc_01 by symmetry
        const float vcc_11 = j11 * a11;
        const float vcc_12 = j11 * a12;
        const float vcc_20 = j02 * a00 + j12 * a10;   // == vcc_02
        const float vcc_21 = j02 * a01 + j12 * a11;   // == vcc_12
        const float vcc_22 = j02 * a02 + j12 * a12;
        // Storage: float3x3(c0, c1, c2) where c_i is column i (= math row i
        // since v_covar_c is sym). Off-diagonals are constructed equal by
        // symmetry; average to absorb fp32 round-off.
        const float vcc_off_01 = 0.5f * (vcc_01 + vcc_10);
        const float vcc_off_02 = 0.5f * (vcc_02 + vcc_20);
        const float vcc_off_12 = 0.5f * (vcc_12 + vcc_21);
        const float3x3 v_covar_c = float3x3(
            float3(vcc_00,     vcc_off_01, vcc_off_02),
            float3(vcc_off_01, vcc_11,     vcc_off_12),
            float3(vcc_off_02, vcc_off_12, vcc_22)
        );

        // Step 3: v_J = 2 * V_2d @ J @ covar_c (only need (0,0),(0,2),(1,1),(1,2)).
        // Reuse a00..a12 = (V_2d @ J): [2x3].
        // v_J = 2 * (V_2d @ J) @ covar_c. covar_c sym 3x3.
        // covar_c_math[i,j] = covar_c_storage[j][i].
        const float c00 = covar_c[0][0], c01 = covar_c[1][0], c02 = covar_c[2][0];
        const float c11 = covar_c[1][1], c12 = covar_c[2][1];
        const float c22 = covar_c[2][2];
        // v_J[r, k] = 2 * (a_r0*c_0k + a_r1*c_1k + a_r2*c_2k).
        const float v_j00 = 2.0f * (a00 * c00 + a01 * c01 + a02 * c02);
        const float v_j02 = 2.0f * (a00 * c02 + a01 * c12 + a02 * c22);
        const float v_j11 = 2.0f * (a10 * c01 + a11 * c11 + a12 * c12);
        const float v_j12 = 2.0f * (a10 * c02 + a11 * c12 + a12 * c22);
        // (v_J[0,1] and v_J[1,0] are 0 in the forward; their v_J contributions
        // back to mean_c are 0 because those slots aren't used.)

        // Step 4: v_J → v_mean_c via dJ/dmean_c.
        // j00 = fx*rz                     ⇒ dj00/dmean_c.z = -fx*rz²
        // j11 = fy*rz                     ⇒ dj11/dmean_c.z = -fy*rz²
        // j02 = -fx*tx*rz²                ⇒ dj02/dtx = -fx*rz²
        //                                   dj02/drz = -2*fx*tx*rz   (drz/dz = -rz²)
        //                                   ⇒ dj02/dmean_c.z = -fx*rz²*(dtx/dmean_c.z) + 2*fx*tx*rz³
        //                                     dj02/dmean_c.x = -fx*rz²*(dtx/dmean_c.x)
        // tx = mean_c.z * xz_c. Cases:
        //   unclamped: xz_c = mean_c.x*rz  → dtx/dmean_c.x = 1, dtx/dmean_c.z = 0
        //   clamped+:  xz_c = lim_x_pos   → dtx/dmean_c.x = 0, dtx/dmean_c.z = lim_x_pos
        //   clamped-:  xz_c = -lim_x_neg  → dtx/dmean_c.x = 0, dtx/dmean_c.z = -lim_x_neg
        const float dtx_dx = (clamp_x_hi || clamp_x_lo) ? 0.0f : 1.0f;
        const float dtx_dz = clamp_x_hi ? lim_x_pos : (clamp_x_lo ? -lim_x_neg : 0.0f);
        const float dty_dy = (clamp_y_hi || clamp_y_lo) ? 0.0f : 1.0f;
        const float dty_dz = clamp_y_hi ? lim_y_pos : (clamp_y_lo ? -lim_y_neg : 0.0f);

        // d j02 / d mean_c.x = -fx*rz²*dtx_dx
        // d j02 / d mean_c.z = -fx*rz²*dtx_dz + 2*fx*tx*rz*rz²  (drz_drz = 1, drz/dz = -rz²)
        //   But careful: we treat tx as the *clamped* product, and rz as its own var.
        //   j02 = -fx * tx * rz² ; d j02 / d rz = -2*fx*tx*rz ; d rz / d mean_c.z = -rz²
        //   So d j02 / d mean_c.z (via rz) = (-2*fx*tx*rz) * (-rz²) = 2*fx*tx*rz³
        //   Plus d j02 / d mean_c.z (via tx) = -fx*rz² * dtx_dz
        const float dj00_dz = -fx * rz2;
        const float dj11_dz = -fy * rz2;
        const float dj02_dx = -fx * rz2 * dtx_dx;
        const float dj02_dz = -fx * rz2 * dtx_dz + 2.0f * fx * tx * rz * rz2;
        const float dj12_dy = -fy * rz2 * dty_dy;
        const float dj12_dz = -fy * rz2 * dty_dz + 2.0f * fy * ty * rz * rz2;

        v_mean_c.x += v_j02 * dj02_dx;
        v_mean_c.y += v_j12 * dj12_dy;
        v_mean_c.z += v_j00 * dj00_dz + v_j02 * dj02_dz
                    + v_j11 * dj11_dz + v_j12 * dj12_dz;

        // Step 5: v_covar_c → v_covar_w  (R_view^T @ V @ R_view)
        const float3x3 RT = transpose(R_view);
        const float3x3 v_covar_w = RT * v_covar_c * R_view;

        // Step 6: v_covar_w → v_M = 2 * v_covar_w @ M.
        const float3x3 v_M = 2.0f * v_covar_w * M;

        // Step 7: v_M → v_R_q  (column j of M = column j of R_q scaled by s[j]).
        //   v_R_q[:, j] = v_M[:, j] * s[j]   (columns)
        //   v_s[j]      = sum_i v_M[i, j] * R_q[i, j]
        // Storage: v_R_q[col=j][row=i] = v_M[col=j][row=i] * s[j]
        const float3 vm_col0 = v_M[0];
        const float3 vm_col1 = v_M[1];
        const float3 vm_col2 = v_M[2];
        const float3 v_Rq_col0 = vm_col0 * s.x;
        const float3 v_Rq_col1 = vm_col1 * s.y;
        const float3 v_Rq_col2 = vm_col2 * s.z;
        const float3 vsv = float3(
            dot(vm_col0, R_q[0]),
            dot(vm_col1, R_q[1]),
            dot(vm_col2, R_q[2])
        );

        // Step 8: v_R_q → v_q_norm.
        // R[i, j] = R_q_math[i, j].
        // v_R_math[i, j] = v_R_q_storage[j][i] (because storage col=j row=i).
        // Apply quat-to-rotmat VJP using the standard formulas.
        // Build access helpers: vR(i, j) reads storage[j][i].
        #define vR(i, j) (v_Rq_col##j[i])
        // Manually inline to avoid dynamic indexing:
        const float vR00 = v_Rq_col0[0];
        const float vR10 = v_Rq_col0[1];
        const float vR20 = v_Rq_col0[2];
        const float vR01 = v_Rq_col1[0];
        const float vR11 = v_Rq_col1[1];
        const float vR21 = v_Rq_col1[2];
        const float vR02 = v_Rq_col2[0];
        const float vR12 = v_Rq_col2[1];
        const float vR22 = v_Rq_col2[2];
        #undef vR

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

        // Step 9: v_q_norm → v_q_raw via L2-normalize VJP.
        // v_q_raw = (v_q_norm - (v_q_norm . q_norm) * q_norm) / |q_raw|.
        const float dot_vqn = vq_w * qw + vq_x * qx + vq_y * qy + vq_z * qz;
        const float vq_w_raw = (vq_w - dot_vqn * qw) * inv_qn;
        const float vq_x_raw = (vq_x - dot_vqn * qx) * inv_qn;
        const float vq_y_raw = (vq_y - dot_vqn * qy) * inv_qn;
        const float vq_z_raw = (vq_z - dot_vqn * qz) * inv_qn;

        // Atomic-add v_quats and v_scales (shared across cameras).
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 0, vq_w_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 1, vq_x_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 2, vq_y_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_quats + mn_off * 4 + 3, vq_z_raw, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 0, vsv.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 1, vsv.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_scales + mn_off * 3 + 2, vsv.z, memory_order_relaxed);
    }

    // Step 10: v_mean_c → v_mean_w  (R_view^T @ v_mean_c).
    const float3x3 RT_view = transpose(R_view);
    const float3 v_mean_w = RT_view * v_mean_c;

    atomic_fetch_add_explicit(v_means + mn_off * 3 + 0, v_mean_w.x, memory_order_relaxed);
    atomic_fetch_add_explicit(v_means + mn_off * 3 + 1, v_mean_w.y, memory_order_relaxed);
    atomic_fetch_add_explicit(v_means + mn_off * 3 + 2, v_mean_w.z, memory_order_relaxed);
}
