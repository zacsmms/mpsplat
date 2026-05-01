// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Metal port of the `intersect_tile` op (CUDA reference:
// `gsplat/cuda/csrc/IntersectTile.cu`). One thread per (image, gaussian)
// writes the tile-bbox-spanning isect entries into pre-allocated buffers.
// The CUB radix sort that follows on CUDA is replaced by `torch.sort` on
// the host (MPS-supported on int64) — see `_intersect_tile_metal`.
//
// Output encoding (matches the torch reference):
//   isect_ids[k] = ((int64)image_id << (tile_n_bits + 32))
//                | ((int64)tile_id  << 32)
//                | (uint64) reinterpret_cast<uint32>(depth_f32)
//
// We split the int64 into two int32 buffers (`isect_ids_lo` = depth bits,
// `isect_ids_hi` = image|tile bits) and merge on the host. This avoids
// MSL int64-alignment surprises and is the same trick the torch reference
// uses internally.

#include <metal_stdlib>
using namespace metal;

kernel void intersect_tile(
    // outputs
    device int*                 isect_ids_lo    [[buffer(0)]], // [n_isects]
    device int*                 isect_ids_hi    [[buffer(1)]], // [n_isects]
    device int*                 flatten_ids     [[buffer(2)]], // [n_isects]
    // inputs (per-(image,gauss) data)
    device const packed_float2* means2d         [[buffer(3)]], // [I*N]
    device const packed_int2*   radii           [[buffer(4)]], // [I*N]
    device const float*         depths          [[buffer(5)]], // [I*N]
    // prefix-summed tiles_per_gauss (size I*N). cum_tiles[k] = total tile
    // entries written by gauss-indices [0..k]. Element k's write offset is
    // cum_tiles[k-1] (or 0 for k=0).
    device const int*           cum_tiles       [[buffer(6)]], // [I*N]
    // shape / bit-layout
    constant uint&              N               [[buffer(7)]],
    constant uint&              tile_size       [[buffer(8)]],
    constant uint&              tile_width      [[buffer(9)]],
    constant uint&              tile_height     [[buffer(10)]],
    constant uint&              tile_n_bits     [[buffer(11)]],
    uint2                       gid             [[thread_position_in_grid]]
) {
    const uint image_id = gid.y;
    const uint gauss_id = gid.x;
    if (gauss_id >= N) {
        return;
    }
    const uint flat_idx = image_id * N + gauss_id;

    const int rx = radii[flat_idx].x;
    const int ry = radii[flat_idx].y;
    if (rx <= 0 || ry <= 0) {
        return;  // no contribution
    }

    const float2 mu = means2d[flat_idx];
    const float ts = (float)tile_size;
    const float tmin_x_raw = floor(mu.x / ts - (float)rx / ts);
    const float tmin_y_raw = floor(mu.y / ts - (float)ry / ts);
    const float tmax_x_raw = ceil (mu.x / ts + (float)rx / ts);
    const float tmax_y_raw = ceil (mu.y / ts + (float)ry / ts);
    int tmin_x = clamp((int)tmin_x_raw, 0, (int)tile_width);
    int tmin_y = clamp((int)tmin_y_raw, 0, (int)tile_height);
    int tmax_x = clamp((int)tmax_x_raw, 0, (int)tile_width);
    int tmax_y = clamp((int)tmax_y_raw, 0, (int)tile_height);
    if (tmax_x <= tmin_x || tmax_y <= tmin_y) {
        return;
    }

    // Reinterpret depth float bits as int32 to make depth comparable as a
    // sortable suffix of the 64-bit isect key. (Positive floats sort the
    // same way as their bit-cast int32 representation.)
    const uint depth_bits = as_type<uint>(depths[flat_idx]);

    // Write offset for this (image, gauss) is the *exclusive* prefix:
    //   cum_tiles[flat_idx - 1] for flat_idx > 0, else 0.
    int curr = (flat_idx == 0u) ? 0 : cum_tiles[flat_idx - 1];

    const int img_shifted = (int)((uint)image_id << tile_n_bits);
    for (int y = tmin_y; y < tmax_y; ++y) {
        const int row_id = y * (int)tile_width;
        for (int x = tmin_x; x < tmax_x; ++x) {
            const int tile_id = row_id + x;
            isect_ids_lo[curr] = (int)depth_bits;
            isect_ids_hi[curr] = img_shifted | tile_id;
            flatten_ids[curr]  = (int)flat_idx;
            ++curr;
        }
    }
}
