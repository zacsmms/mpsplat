# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed 2DGS forward + backward rasterizer.

The forward kernel mirrors `rasterize_to_pixels_3dgs_fwd` but uses the 2DGS
ray-splat alpha formulation (see `_torch_impl_2dgs.accumulate_2dgs`). The
backward kernel walks the same gaussians in reverse, atomic-scatters
gradients into per-gaussian buffers, and routes the sigma gradient into
either `means2d` or `ray_transforms` depending on which sigma branch
(2D vs 3D ray-splat) the forward selected.
"""

from typing import Tuple

import torch
from torch import Tensor

from ._kernels import _load


_TILE_SIZE = 16  # must match TILE_SIZE in rasterize_to_pixels_2dgs_fwd.metal


def _rasterize_to_pixels_2dgs_fwd_metal(
    means2d: Tensor,  # [I, N, 2]
    ray_transforms: Tensor,  # [I, N, 3, 3]
    colors: Tensor,  # [I, N, 3]
    opacities: Tensor,  # [I, N]
    normals: Tensor,  # [I, N, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,  # [I, tile_h, tile_w]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns (render_colors, render_alphas, render_normals, last_ids)."""
    assert tile_size == _TILE_SIZE, f"only tile_size={_TILE_SIZE} is supported"
    assert colors.shape[-1] == 3, "Metal 2DGS fwd hard-codes CDIM=3"
    assert means2d.dim() == 3, "packed mode not supported in Metal forward yet"

    I, N = means2d.shape[0], means2d.shape[1]
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]
    assert tile_offsets.shape == (I, tile_h, tile_w)

    means2d_flat = means2d.contiguous().reshape(I * N, 2)
    ray_transforms_flat = ray_transforms.contiguous().reshape(I * N, 9)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    opacities_flat = opacities.contiguous().reshape(I * N)
    normals_flat = normals.contiguous().reshape(I * N, 3)
    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()

    device = means2d.device
    render_colors = torch.zeros(I, H, W, 3, device=device, dtype=torch.float32)
    render_alphas = torch.zeros(I, H, W, 1, device=device, dtype=torch.float32)
    render_normals = torch.zeros(I, H, W, 3, device=device, dtype=torch.float32)
    last_ids = torch.zeros(I, H, W, device=device, dtype=torch.int32)

    n_isects = int(flatten_ids.numel())
    lib = _load("rasterize_to_pixels_2dgs_fwd")
    lib.rasterize_to_pixels_2dgs_fwd(
        render_colors,
        render_alphas,
        render_normals,
        last_ids,
        means2d_flat,
        ray_transforms_flat,
        colors_flat,
        opacities_flat,
        normals_flat,
        tile_offsets_i32,
        flatten_ids_i32,
        W,
        H,
        tile_w,
        tile_h,
        n_isects,
        I,
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )
    return render_colors, render_alphas, render_normals, last_ids


def _rasterize_to_pixels_2dgs_bwd_metal(
    means2d: Tensor,  # [I, N, 2]
    ray_transforms: Tensor,  # [I, N, 3, 3]
    colors: Tensor,  # [I, N, 3]
    opacities: Tensor,  # [I, N]
    normals: Tensor,  # [I, N, 3]
    tile_offsets: Tensor,  # [I, tile_h, tile_w]
    flatten_ids: Tensor,
    render_alphas: Tensor,  # [I, H, W, 1]
    last_ids: Tensor,  # [I, H, W]
    v_render_colors: Tensor,  # [I, H, W, 3]
    v_render_alphas: Tensor,  # [I, H, W, 1]
    v_render_normals: Tensor,  # [I, H, W, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns dense [I, N, ...] grads for means2d, ray_transforms, colors, opacities, normals."""
    assert tile_size == _TILE_SIZE
    assert colors.shape[-1] == 3
    I, N = means2d.shape[0], means2d.shape[1]
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]

    means2d_flat = means2d.contiguous().reshape(I * N, 2)
    ray_transforms_flat = ray_transforms.contiguous().reshape(I * N, 9)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    opacities_flat = opacities.contiguous().reshape(I * N)
    normals_flat = normals.contiguous().reshape(I * N, 3)
    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()
    render_alphas_flat = render_alphas.contiguous().reshape(I * H * W)
    last_ids_flat = last_ids.to(torch.int32).contiguous().reshape(I * H * W)
    v_render_colors_flat = v_render_colors.contiguous().reshape(I * H * W * 3)
    v_render_alphas_flat = v_render_alphas.contiguous().reshape(I * H * W)
    v_render_normals_flat = v_render_normals.contiguous().reshape(I * H * W * 3)

    device = means2d.device
    v_means2d = torch.zeros(I * N, 2, device=device, dtype=torch.float32)
    v_ray_transforms = torch.zeros(I * N, 9, device=device, dtype=torch.float32)
    v_colors = torch.zeros(I * N, 3, device=device, dtype=torch.float32)
    v_opacities = torch.zeros(I * N, device=device, dtype=torch.float32)
    v_normals = torch.zeros(I * N, 3, device=device, dtype=torch.float32)

    n_isects = int(flatten_ids.numel())
    lib = _load("rasterize_to_pixels_2dgs_bwd")
    lib.rasterize_to_pixels_2dgs_bwd(
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        means2d_flat,
        ray_transforms_flat,
        colors_flat,
        opacities_flat,
        normals_flat,
        render_alphas_flat,
        last_ids_flat,
        v_render_colors_flat,
        v_render_alphas_flat,
        v_render_normals_flat,
        tile_offsets_i32,
        flatten_ids_i32,
        W,
        H,
        tile_w,
        tile_h,
        n_isects,
        I,
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )
    return (
        v_means2d.reshape(I, N, 2),
        v_ray_transforms.reshape(I, N, 3, 3),
        v_colors.reshape(I, N, 3),
        v_opacities.reshape(I, N),
        v_normals.reshape(I, N, 3),
    )


class _RasterizeToPixels2DGSMetal(torch.autograd.Function):
    """2DGS rasterizer with Metal forward + Metal backward."""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        tile_offsets: Tensor,
        flatten_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        rc, ra, rn, last_ids = _rasterize_to_pixels_2dgs_fwd_metal(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
        )
        ctx.save_for_backward(
            means2d, ray_transforms, colors, opacities, normals,
            tile_offsets, flatten_ids, ra, last_ids,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.tile_size = tile_size
        return rc, ra, rn

    @staticmethod
    def backward(ctx, grad_colors: Tensor, grad_alphas: Tensor, grad_normals: Tensor):
        (
            means2d, ray_transforms, colors, opacities, normals,
            tile_offsets, flatten_ids, render_alphas, last_ids,
        ) = ctx.saved_tensors
        v_means2d, v_rt, v_colors, v_opacities, v_normals = (
            _rasterize_to_pixels_2dgs_bwd_metal(
                means2d, ray_transforms, colors, opacities, normals,
                tile_offsets, flatten_ids,
                render_alphas, last_ids,
                grad_colors, grad_alphas, grad_normals,
                ctx.image_width, ctx.image_height, ctx.tile_size,
            )
        )
        return (
            v_means2d, v_rt, v_colors, v_opacities, v_normals,
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # tile_offsets
            None,  # flatten_ids
        )


def rasterize_to_pixels_2dgs_metal(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Public-facing entry point. Returns (colors, alphas, normals)."""
    return _RasterizeToPixels2DGSMetal.apply(
        means2d, ray_transforms, colors, opacities, normals,
        image_width, image_height, tile_size,
        tile_offsets, flatten_ids,
    )


__all__ = ["rasterize_to_pixels_2dgs_metal"]
