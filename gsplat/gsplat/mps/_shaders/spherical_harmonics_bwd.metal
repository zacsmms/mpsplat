// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Spherical harmonics backward pass.
//
// Forward (in `spherical_harmonics_fwd.metal`):
//   d_norm := dirs / |dirs|
//   b[k]   := SH_k(d_norm.x, d_norm.y, d_norm.z)         k in [0, K)
//   out[c] := sum_k b[k] * coeffs[k, c]                  c in {R, G, B}
//
// Backward (this kernel):
//   v_coeffs[k, c] = b[k] * v_out[c]                     // scatter (no atomic; one thread per m)
//   v_b[k]         = sum_c coeffs[k, c] * v_out[c]
//   v_dn[a]        = sum_k v_b[k] * dY_k/d(d_norm[a])     a in {x,y,z}
//   v_dirs         = (v_dn - dot(v_dn, d_norm) * d_norm) / |dirs|   // L2-normalize VJP
//
// Each basis Y_k = polynomial in d_norm; gradients dY_k/d(x,y,z) are
// hard-coded inline. Up to degree 4 (K <= 25), matching the forward kernel.

#include <metal_stdlib>
using namespace metal;

kernel void spherical_harmonics_bwd(
    device float*        v_dirs       [[buffer(0)]],   // [M, 3]
    device float*        v_coeffs     [[buffer(1)]],   // [M, K, 3]
    device const float*  dirs         [[buffer(2)]],   // [M, 3]
    device const float*  coeffs       [[buffer(3)]],   // [M, K, 3]
    device const float*  v_out        [[buffer(4)]],   // [M, 3]
    constant uint&       M            [[buffer(5)]],
    constant uint&       K            [[buffer(6)]],
    constant uint&       degrees      [[buffer(7)]],
    uint                 idx          [[thread_position_in_grid]]
) {
    if (idx >= M) {
        return;
    }

    const float3 d_raw = float3(
        dirs[idx * 3 + 0], dirs[idx * 3 + 1], dirs[idx * 3 + 2]);
    const float n_raw = max(length(d_raw), 1e-12f);
    const float3 d = d_raw / n_raw;
    const float x = d.x, y = d.y, z = d.z;

    const float3 v = float3(
        v_out[idx * 3 + 0], v_out[idx * 3 + 1], v_out[idx * 3 + 2]);

    float b[25];
    float gx[25];
    float gy[25];
    float gz[25];
    for (uint k = 0; k < 25; ++k) {
        b[k] = 0.0f; gx[k] = 0.0f; gy[k] = 0.0f; gz[k] = 0.0f;
    }

    // l = 0
    b[0] = 0.2820947917738781f;

    // l = 1
    const float fTmpA1 = -0.48860251190292f;
    if (degrees >= 1) {
        b[2] = -fTmpA1 * z;   gz[2] = -fTmpA1;
        b[3] =  fTmpA1 * x;   gx[3] =  fTmpA1;
        b[1] =  fTmpA1 * y;   gy[1] =  fTmpA1;
    }

    // l = 2
    float z2 = 0.0f, fTmpA2 = 0.0f, fTmpB2 = 0.0f, fC1 = 0.0f, fS1 = 0.0f;
    if (degrees >= 2) {
        z2 = z * z;
        fTmpB2 = -1.092548430592079f * z;
        fTmpA2 = 0.5462742152960395f;
        fC1 = x * x - y * y;
        fS1 = 2.0f * x * y;

        b[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
        gz[6] = 2.0f * 0.9461746957575601f * z;

        b[7] = fTmpB2 * x;
        gx[7] = fTmpB2;
        gz[7] = -1.092548430592079f * x;

        b[5] = fTmpB2 * y;
        gy[5] = fTmpB2;
        gz[5] = -1.092548430592079f * y;

        b[8] = fTmpA2 * fC1;
        gx[8] =  2.0f * fTmpA2 * x;
        gy[8] = -2.0f * fTmpA2 * y;

        b[4] = fTmpA2 * fS1;
        gx[4] = 2.0f * fTmpA2 * y;
        gy[4] = 2.0f * fTmpA2 * x;
    }

    // l = 3
    float fC2 = 0.0f, fS2 = 0.0f;
    if (degrees >= 3) {
        const float fTmpC3 = -2.285228997322329f * z2 + 0.4570457994644658f;
        const float fTmpB3 =  1.445305721320277f * z;
        const float fTmpA3 = -0.5900435899266435f;
        fC2 = x * fC1 - y * fS1;
        fS2 = x * fS1 + y * fC1;

        b[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
        gz[12] = 3.0f * 1.865881662950577f * z2 - 1.119528997770346f;

        b[13] = fTmpC3 * x;
        gx[13] = fTmpC3;
        gz[13] = -2.0f * 2.285228997322329f * z * x;

        b[11] = fTmpC3 * y;
        gy[11] = fTmpC3;
        gz[11] = -2.0f * 2.285228997322329f * z * y;

        b[14] = fTmpB3 * fC1;
        gx[14] =  2.0f * fTmpB3 * x;
        gy[14] = -2.0f * fTmpB3 * y;
        gz[14] = 1.445305721320277f * fC1;

        b[10] = fTmpB3 * fS1;
        gx[10] = 2.0f * fTmpB3 * y;
        gy[10] = 2.0f * fTmpB3 * x;
        gz[10] = 1.445305721320277f * fS1;

        b[15] = fTmpA3 * fC2;
        gx[15] =  3.0f * fTmpA3 * fC1;
        gy[15] = -3.0f * fTmpA3 * fS1;

        b[9]  = fTmpA3 * fS2;
        gx[9] = 3.0f * fTmpA3 * fS1;
        gy[9] = 3.0f * fTmpA3 * fC1;
    }

    // l = 4
    if (degrees >= 4) {
        const float fTmpD  = z * (-4.683325804901025f * z2 + 2.007139630671868f);
        const float fTmpC4 = 3.31161143515146f * z2 - 0.47308734787878f;
        const float fTmpB4 = -1.770130769779931f * z;
        const float fTmpA4 = 0.6258357354491763f;
        const float fC3 = x * fC2 - y * fS2;
        const float fS3 = x * fS2 + y * fC2;

        b[20] = 1.984313483298443f * z2 *
                    (1.865881662950577f * z2 - 1.119528997770346f)
              + -1.006230589874905f *
                    (0.9461746957575601f * z2 - 0.3153915652525201f);
        // d b[20]/dz = 1.984... * [2z*(1.865 z^2 - 1.119) + z^2 * 2*1.865 z]
        //             + (-1.006...) * 2*0.9461... z
        gz[20] =
            1.984313483298443f * (
                2.0f * z * (1.865881662950577f * z2 - 1.119528997770346f)
              + z2 * 2.0f * 1.865881662950577f * z)
          + -1.006230589874905f * 2.0f * 0.9461746957575601f * z;

        b[21] = fTmpD * x;
        gx[21] = fTmpD;
        // dfTmpD/dz = -3*4.683...*z^2 + 2.007...
        gz[21] = (-3.0f * 4.683325804901025f * z2 + 2.007139630671868f) * x;

        b[19] = fTmpD * y;
        gy[19] = fTmpD;
        gz[19] = (-3.0f * 4.683325804901025f * z2 + 2.007139630671868f) * y;

        b[22] = fTmpC4 * fC1;
        gx[22] =  2.0f * fTmpC4 * x;
        gy[22] = -2.0f * fTmpC4 * y;
        // dfTmpC4/dz = 2*3.311... z
        gz[22] = 2.0f * 3.31161143515146f * z * fC1;

        b[18] = fTmpC4 * fS1;
        gx[18] = 2.0f * fTmpC4 * y;
        gy[18] = 2.0f * fTmpC4 * x;
        gz[18] = 2.0f * 3.31161143515146f * z * fS1;

        b[23] = fTmpB4 * fC2;
        gx[23] =  3.0f * fTmpB4 * fC1;
        gy[23] = -3.0f * fTmpB4 * fS1;
        gz[23] = -1.770130769779931f * fC2;

        b[17] = fTmpB4 * fS2;
        gx[17] = 3.0f * fTmpB4 * fS1;
        gy[17] = 3.0f * fTmpB4 * fC1;
        gz[17] = -1.770130769779931f * fS2;

        // d fC3 / dx = 4*fC2,  d fC3 / dy = -4*fS2
        b[24] = fTmpA4 * fC3;
        gx[24] =  4.0f * fTmpA4 * fC2;
        gy[24] = -4.0f * fTmpA4 * fS2;

        // d fS3 / dx = 4*fS2,  d fS3 / dy = 4*fC2
        b[16] = fTmpA4 * fS3;
        gx[16] = 4.0f * fTmpA4 * fS2;
        gy[16] = 4.0f * fTmpA4 * fC2;
    }

    // Scatter v_coeffs and accumulate v_dn.
    float v_dn_x = 0.0f, v_dn_y = 0.0f, v_dn_z = 0.0f;
    const uint coeff_off = idx * K * 3u;
    for (uint k = 0; k < K; ++k) {
        const float bk = b[k];
        const float c0 = coeffs[coeff_off + k * 3 + 0];
        const float c1 = coeffs[coeff_off + k * 3 + 1];
        const float c2 = coeffs[coeff_off + k * 3 + 2];

        // v_coeffs[k, c] = b[k] * v_out[c]
        v_coeffs[coeff_off + k * 3 + 0] = bk * v.x;
        v_coeffs[coeff_off + k * 3 + 1] = bk * v.y;
        v_coeffs[coeff_off + k * 3 + 2] = bk * v.z;

        // v_b[k] = sum_c coeffs[k, c] * v_out[c]
        const float vbk = c0 * v.x + c1 * v.y + c2 * v.z;
        v_dn_x += vbk * gx[k];
        v_dn_y += vbk * gy[k];
        v_dn_z += vbk * gz[k];
    }

    // Chain through the L2 normalize: v_dirs = (v_dn - (v_dn . d) * d) / |dirs|
    const float dot_vd = v_dn_x * x + v_dn_y * y + v_dn_z * z;
    const float inv_n = 1.0f / n_raw;
    v_dirs[idx * 3 + 0] = (v_dn_x - dot_vd * x) * inv_n;
    v_dirs[idx * 3 + 1] = (v_dn_y - dot_vd * y) * inv_n;
    v_dirs[idx * 3 + 2] = (v_dn_z - dot_vd * z) * inv_n;
}
