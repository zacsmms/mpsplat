// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Spherical harmonics evaluation (forward) — port of
// `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`. The basis evaluation order
// matches `_torch_impl._eval_sh_bases_fast` exactly; the only thing that
// changes per-degree is how many bases get evaluated.
//
// One thread per output element. Each thread reads its 3D direction,
// normalises it, evaluates the SH basis vector (size K = (deg+1)^2),
// then dot-products with the per-channel coefficient buffer to produce
// the output color (RGB).

#include <metal_stdlib>
using namespace metal;

inline float sh0_const() { return 0.2820947917738781f; }

kernel void spherical_harmonics_fwd(
    device float*        out          [[buffer(0)]],   // [M, 3]
    device const float*  dirs         [[buffer(1)]],   // [M, 3]
    device const float*  coeffs       [[buffer(2)]],   // [M, K, 3]
    constant uint&       M            [[buffer(3)]],
    constant uint&       K            [[buffer(4)]],
    constant uint&       degrees      [[buffer(5)]],
    uint                 idx          [[thread_position_in_grid]]
) {
    if (idx >= M) {
        return;
    }

    // L2-normalize the direction.
    float3 d = float3(dirs[idx * 3 + 0], dirs[idx * 3 + 1], dirs[idx * 3 + 2]);
    const float n = max(length(d), 1e-12f);
    d /= n;
    const float x = d.x, y = d.y, z = d.z;

    // SH bases up to (degrees+1)^2 — values match `_eval_sh_bases_fast`.
    float b[25];
    for (uint k = 0; k < 25; ++k) b[k] = 0.0f;

    b[0] = sh0_const();

    if (degrees >= 1) {
        const float fTmpA1 = -0.48860251190292f;
        b[2] = -fTmpA1 * z;
        b[3] =  fTmpA1 * x;
        b[1] =  fTmpA1 * y;
    }

    if (degrees >= 2) {
        const float z2 = z * z;
        const float fTmpB = -1.092548430592079f * z;
        const float fTmpA = 0.5462742152960395f;
        const float fC1 = x * x - y * y;
        const float fS1 = 2.0f * x * y;
        b[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
        b[7] = fTmpB * x;
        b[5] = fTmpB * y;
        b[8] = fTmpA * fC1;
        b[4] = fTmpA * fS1;

        if (degrees >= 3) {
            const float fTmpC = -2.285228997322329f * z2 + 0.4570457994644658f;
            const float fTmpB3 = 1.445305721320277f * z;
            const float fTmpA3 = -0.5900435899266435f;
            const float fC2 = x * fC1 - y * fS1;
            const float fS2 = x * fS1 + y * fC1;
            b[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
            b[13] = fTmpC * x;
            b[11] = fTmpC * y;
            b[14] = fTmpB3 * fC1;
            b[10] = fTmpB3 * fS1;
            b[15] = fTmpA3 * fC2;
            b[9]  = fTmpA3 * fS2;

            if (degrees >= 4) {
                const float fTmpD = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                const float fTmpC4 = 3.31161143515146f * z2 - 0.47308734787878f;
                const float fTmpB4 = -1.770130769779931f * z;
                const float fTmpA4 = 0.6258357354491763f;
                const float fC3 = x * fC2 - y * fS2;
                const float fS3 = x * fS2 + y * fC2;
                b[20] = 1.984313483298443f * z2 *
                            (1.865881662950577f * z2 - 1.119528997770346f)
                      + -1.006230589874905f *
                            (0.9461746957575601f * z2 - 0.3153915652525201f);
                b[21] = fTmpD * x;
                b[19] = fTmpD * y;
                b[22] = fTmpC4 * fC1;
                b[18] = fTmpC4 * fS1;
                b[23] = fTmpB4 * fC2;
                b[17] = fTmpB4 * fS2;
                b[24] = fTmpA4 * fC3;
                b[16] = fTmpA4 * fS3;
            }
        }
    }

    // Dot bases with coeffs, per RGB channel.
    float rgb_r = 0.0f, rgb_g = 0.0f, rgb_b = 0.0f;
    const uint coeff_off = idx * K * 3u;
    for (uint k = 0; k < K; ++k) {
        const float bk = b[k];
        rgb_r += bk * coeffs[coeff_off + k * 3 + 0];
        rgb_g += bk * coeffs[coeff_off + k * 3 + 1];
        rgb_b += bk * coeffs[coeff_off + k * 3 + 2];
    }
    out[idx * 3 + 0] = rgb_r;
    out[idx * 3 + 1] = rgb_g;
    out[idx * 3 + 2] = rgb_b;
}
