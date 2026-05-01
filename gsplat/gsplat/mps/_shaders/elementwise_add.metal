// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Toy elementwise-add shader. Used by tests/test_mps_metal.py to verify the
// torch.mps.compile_shader build pipeline works end-to-end. Phase B (Stages
// 6+) replaces this with the real rasterizer / projection / SH kernels.

#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device float*       out [[buffer(0)]],
    device const float* a   [[buffer(1)]],
    device const float* b   [[buffer(2)]],
    uint                tid [[thread_position_in_grid]]
) {
    out[tid] = a[tid] + b[tid];
}
