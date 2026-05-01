# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed `intersect_tile` (Stage 8).

The CUDA backend used CUB radix sort to order isect entries by
(image_id, tile_id, depth). On MPS we use a Metal kernel to populate the
buffers and `torch.sort` (stable) for the sort itself — `torch.sort` is
MPS-supported on int64 and is far simpler to maintain than a custom Metal
radix sort. Performance is sufficient for typical scene sizes; if profiling
ever shows the sort as the bottleneck, a Metal radix sort can land here.
"""

from typing import Tuple

import torch
from torch import Tensor

from ._kernels import _load


def _intersect_tile_metal(
    means2d: Tensor,  # [I, N, 2]  fp32, mps
    radii: Tensor,  # [I, N, 2]   int32, mps
    depths: Tensor,  # [I, N]     fp32, mps
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Drop-in replacement for `_torch_impl._isect_tiles` on MPS.

    Returns `(tiles_per_gauss, isect_ids, flatten_ids)`.
    """
    assert means2d.device.type == "mps"
    image_dims = means2d.shape[:-2]
    N = means2d.shape[-2]
    I = 1
    for d in image_dims:
        I *= int(d)
    means2d_f = means2d.contiguous().reshape(I, N, 2)
    radii_f = radii.to(torch.int32).contiguous().reshape(I, N, 2)
    depths_f = depths.contiguous().reshape(I, N)

    # tiles_per_gauss can be computed entirely in torch — fast on MPS.
    tile_means2d = means2d_f / tile_size
    tile_radii = radii_f / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = tile_mins[..., 0].clamp(0, tile_width)
    tile_mins[..., 1] = tile_mins[..., 1].clamp(0, tile_height)
    tile_maxs[..., 0] = tile_maxs[..., 0].clamp(0, tile_width)
    tile_maxs[..., 1] = tile_maxs[..., 1].clamp(0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)
    tiles_per_gauss = tiles_per_gauss * (radii_f > 0).all(dim=-1).int()
    n_isects = int(tiles_per_gauss.sum().item())

    # Bit layout for the isect key (must match `_torch_impl._isect_tiles`).
    image_n_bits = max(1, I).bit_length()
    tile_n_bits = (tile_width * tile_height).bit_length()
    assert image_n_bits + tile_n_bits + 32 <= 64, "isect key would overflow 64-bit"

    # Inclusive prefix sum over (I*N) for write offsets. Element k's start
    # is cum_tiles[k-1]; we pass the inclusive cumsum and the kernel adjusts.
    cum_tiles = torch.cumsum(tiles_per_gauss.flatten().to(torch.int32), dim=0).to(torch.int32)

    isect_ids_lo = torch.zeros(n_isects, dtype=torch.int32, device=means2d.device)
    isect_ids_hi = torch.zeros(n_isects, dtype=torch.int32, device=means2d.device)
    flatten_ids = torch.zeros(n_isects, dtype=torch.int32, device=means2d.device)

    if n_isects > 0:
        lib = _load("intersect_tile")
        lib.intersect_tile(
            isect_ids_lo,
            isect_ids_hi,
            flatten_ids,
            means2d_f.contiguous().reshape(I * N, 2),
            radii_f.contiguous().reshape(I * N, 2),
            depths_f.contiguous().reshape(I * N),
            cum_tiles.contiguous(),
            N,
            tile_size,
            tile_width,
            tile_height,
            tile_n_bits,
            threads=(N, I),
            group_size=(min(N, 256), 1),
        )

    # Merge the two int32 halves into the int64 isect key. We reinterpret
    # the lo half as uint32 by masking after the cast, matching the torch
    # reference's `(lo & 0xFFFFFFFF)` pattern.
    isect_ids = (isect_ids_hi.to(torch.int64) << 32) | (
        isect_ids_lo.to(torch.int64) & 0xFFFFFFFF
    )

    if sort and n_isects > 0:
        sorted_keys, sort_idx = torch.sort(isect_ids, stable=True)
        isect_ids = sorted_keys
        flatten_ids = flatten_ids[sort_idx]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (N,)).int()
    return tiles_per_gauss, isect_ids, flatten_ids


__all__ = ["_intersect_tile_metal"]
