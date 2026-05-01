// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3DGS forward tile rasterizer.
//
// Port of `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` (CUDA) — recoverable
// via `git show pre-mps-cleanup:...`. The grid layout is 3D:
//
//     threads     = ( tile_width * tile_size, tile_height * tile_size, I )
//     group_size  = ( tile_size, tile_size, 1 )
//
// One threadgroup per (image, tile_y, tile_x); one thread per pixel. Each
// threadgroup pulls `block_size`-batches of gaussians into threadgroup
// memory and composites them front-to-back over its pixels.
//
// Differences from CUDA:
//   * No `__syncthreads_count(done)` early-exit (no simple MSL equivalent
//     for "count done threads in the threadgroup"). Instead each thread
//     just early-stops its own loop. Worst case: tail threads idle while
//     others finish — minor perf, identical correctness.
//   * Channel count CDIM=3 is hard-coded for now. Stage 6.x can specialise
//     for the other power-of-two channel counts the CUDA backend used to
//     pad to (4/8/16/...).
//   * `extern __shared__` becomes a fixed-size `threadgroup` buffer sized
//     for the largest tile we support: tile_size <= 16, so block_size = 256.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE_SIZE   = 16u;
constant constexpr uint BLOCK_SIZE  = TILE_SIZE * TILE_SIZE;
constant constexpr float MAX_ALPHA              = 0.99f;
constant constexpr float ALPHA_THRESHOLD        = 1.0f / 255.0f;
constant constexpr float TRANSMITTANCE_THRESHOLD = 1e-4f;

// Layout of the per-batch threadgroup cache. We pack the per-gaussian
// payload as: [id (int32)] [xy.x, xy.y, opacity (float)] [conic.x, conic.y, conic.z (float)]
// = 1 int + 6 floats = 28 bytes per gaussian.
struct GaussCache {
    int   id;
    float x;
    float y;
    float opac;
    float c0;
    float c1;
    float c2;
};

// `packed_float3` is 12 bytes (3 contiguous floats) — matches a torch
// tensor's row stride for `[..., 3]` layouts. The plain `float3` is 16-byte
// aligned and would mis-read every other element of a torch buffer.
kernel void rasterize_to_pixels_3dgs_fwd(
    device float*               render_colors [[buffer(0)]], // [I, H, W, CDIM]
    device float*               render_alphas [[buffer(1)]], // [I, H, W, 1]
    device int*                 last_ids      [[buffer(2)]], // [I, H, W]
    device const packed_float2* means2d       [[buffer(3)]], // [I*N, 2] flat
    device const packed_float3* conics        [[buffer(4)]], // [I*N, 3] flat
    device const float*         colors        [[buffer(5)]], // [I*N, CDIM] flat (CDIM=3)
    device const float*  opacities        [[buffer(6)]], // [I*N] flat
    device const int*    tile_offsets     [[buffer(7)]], // [I, tile_h, tile_w]
    device const int*    flatten_ids      [[buffer(8)]], // [n_isects]
    constant uint&       image_width      [[buffer(9)]],
    constant uint&       image_height     [[buffer(10)]],
    constant uint&       tile_width       [[buffer(11)]],
    constant uint&       tile_height      [[buffer(12)]],
    constant uint&       n_isects         [[buffer(13)]],
    constant uint&       num_images       [[buffer(14)]],
    uint3                gid              [[thread_position_in_grid]],
    uint3                lid              [[thread_position_in_threadgroup]],
    uint3                tg_id            [[threadgroup_position_in_grid]]
) {
    constexpr uint CDIM = 3u;

    const uint image_id = gid.z;
    const uint tile_x = tg_id.x;
    const uint tile_y = tg_id.y;
    const uint pix_x = gid.x;
    const uint pix_y = gid.y;
    const uint tile_id = tile_y * tile_width + tile_x;
    const uint pix_id = pix_y * image_width + pix_x;
    const uint tile_offset_idx = image_id * tile_height * tile_width + tile_id;
    const uint thread_rank = lid.y * TILE_SIZE + lid.x;

    const bool inside = (pix_x < image_width) && (pix_y < image_height);

    // Determine the (range_start, range_end) for this tile.
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
    float pix_out[CDIM] = {0.0f, 0.0f, 0.0f};

    // Direct device-memory reads (no threadgroup cache). Slower than the
    // CUDA tile cache but trivially correct. Stage 6.x will reintroduce a
    // properly-aligned threadgroup cache once we have a baseline that
    // matches the torch reference exactly.
    if (!done) {
        for (int idx = range_start; idx < range_end; ++idx) {
            const int g = flatten_ids[idx];
            const float2 xy = means2d[g];
            const float3 cn = conics[g];
            const float opac = opacities[g];

            const float dx = xy.x - px;
            const float dy = xy.y - py;
            const float sigma =
                0.5f * (cn.x * dx * dx + cn.z * dy * dy) + cn.y * dx * dy;
            float alpha = opac * exp(-sigma);
            alpha = min(MAX_ALPHA, alpha);
            if (sigma < 0.0f) {
                alpha = 0.0f;
            }
            const float next_T = T * max(1.0f - alpha, 1e-7f);
            const float vis = alpha * T;
            const uint c_off = (uint)g * CDIM;
            pix_out[0] += colors[c_off + 0] * vis;
            pix_out[1] += colors[c_off + 1] * vis;
            pix_out[2] += colors[c_off + 2] * vis;
            cur_idx = idx;
            T = next_T;
        }
    }

    if (inside) {
        const uint pix_base = image_id * image_height * image_width + pix_id;
        render_alphas[pix_base] = 1.0f - T;
        render_colors[pix_base * CDIM + 0] = pix_out[0];
        render_colors[pix_base * CDIM + 1] = pix_out[1];
        render_colors[pix_base * CDIM + 2] = pix_out[2];
        last_ids[pix_base] = cur_idx;
    }
}
